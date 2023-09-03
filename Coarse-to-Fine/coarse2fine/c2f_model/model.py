import torch
import torch.nn as nn

from .layers import transformer
from .layers import utils


class LayoutEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.label_embed = nn.Embedding(cfg.num_labels+3, 128)
        self.bbox_embed = nn.Embedding(max(cfg.discrete_x_grid, cfg.discrete_y_grid), 128)
        self.proj_cat = nn.Linear(128*5, cfg.d_model)

        self.group_label_embed = nn.Linear(cfg.num_labels+2, 128)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.label_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.bbox_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.proj_cat.weight, mode="fan_in")

    def get_label_embedding(self, label):
        return self.label_embed(label)

    def get_box_embedding(self, box):
        bbox_vecs = self.bbox_embed(box)
        S, N, _, E = bbox_vecs.shape
        bbox_vecs = bbox_vecs.reshape(S, N, -1)
        return bbox_vecs

    def get_group_label_embedding(self, label):
        return self.group_label_embed(label)

    def forward(self, label, box):
        label_vecs = self.get_label_embedding(label)
        box_vecs = self.get_box_embedding(box)
        src = self.proj_cat(torch.cat((label_vecs, box_vecs), -1))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg, layout_embd):
        super().__init__()
        self.embedding = layout_embd
        encoder_layer = transformer.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = transformer.LayerNorm(cfg.d_model)
        self.encoder = transformer.TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, labels, bboxes, masks):

        key_padding_mask = utils._get_key_padding_mask(masks)
        src = self.embedding(labels, bboxes)
        memory = self.encoder(src=src, memory2=None, src_key_padding_mask=key_padding_mask)
        padding_mask = utils._get_padding_mask(masks)
        memory = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True)

        return memory


class VAE(nn.Module):
    def __init__(self, cfg):
        super(VAE, self).__init__()

        self.cfg = cfg
        self.enc_mu_fcn = nn.Linear(cfg.d_model, cfg.d_z)
        self.enc_sigma_fcn = nn.Linear(cfg.d_model, cfg.d_z)
        self.z_fcn = nn.Linear(cfg.d_z, cfg.d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, memory):
        mu, logvar = self.enc_mu_fcn(memory), self.enc_sigma_fcn(memory)
        sigma = torch.exp(logvar / 2.)
        z = mu + sigma * torch.randn_like(sigma)
        return z, mu, logvar

    def inference(self, z, device):
        if z is None:
            return torch.randn(size=[1, self.cfg.eval_batch_size, self.cfg.d_z]).to(device)
        else:
            return utils._make_seq_first(z).to(device)


class GroupDecoder(nn.Module):
    def __init__(self, cfg, layout_embd):
        super(GroupDecoder, self).__init__()

        self.cfg = cfg
        self.layout_embd = layout_embd
        self.proj_cat_tgt = nn.Linear(128*5, cfg.d_model)

        square_subsequent_mask = utils._generate_square_subsequent_mask(cfg.max_num_elements+2)
        self.register_buffer("square_subsequent_mask", square_subsequent_mask)

        decoder_layer = transformer.TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = transformer.LayerNorm(cfg.d_model)
        self.decoder = transformer.TransformerDecoder(decoder_layer, cfg.n_layers_decoder, decoder_norm)

        self.label_fcn = nn.Sequential(nn.Linear(cfg.d_model, 128),
                                       nn.Linear(128, cfg.num_labels+2),
                                       nn.ReLU(inplace=True),)
        self.box_fcn = nn.Sequential(nn.Linear(cfg.d_model, 4*max(cfg.discrete_x_grid, cfg.discrete_y_grid)),
                                     nn.ReLU(inplace=True))

    def forward(self, label, box, z, mask):
        '''
        autoregressive decode group bounding box and label in one group
        label(G,N,_), box(G,N,4*_), z(1,N,E), mask(N,G)
        '''
        key_padding_mask = utils._get_key_padding_mask(mask)
        tgt_label_vecs = self.layout_embd.get_group_label_embedding(label)  # (G,N,128)
        tgt_box_vecs = self.layout_embd.get_box_embedding(box)  # (G,N,4*128)
        tgt = self.proj_cat_tgt(torch.cat((tgt_label_vecs, tgt_box_vecs), -1))

        G = tgt.size(0)
        out = self.decoder(tgt[:G], z, tgt_mask=self.square_subsequent_mask[:G, :G], tgt_key_padding_mask=key_padding_mask)
        rec_box = self.box_fcn(out)
        rec_label = self.label_fcn(out)

        out = out[:-2]  # remove eos group embedding
        return out, rec_box, rec_label  # (G,N,E) / (G,N,4*_) / (G,N,_)

    def inference(self, z, device, max_group_num):
        tgt = torch.zeros(max_group_num, z.shape[1], self.cfg.d_model).to(device)  # (G,N,E)
        rec_labels = torch.zeros(max_group_num, z.shape[1], self.cfg.num_labels+2).to(device)  # (G,N,_)
        rec_bboxes = torch.zeros(max_group_num, z.shape[1], 4*max(self.cfg.discrete_x_grid, self.cfg.discrete_y_grid)).to(device)  # (G,N,4*_)

        sos_box = torch.zeros(z.shape[1], 4).to(device).long()
        sos_label = torch.zeros(z.shape[1], self.cfg.num_labels+2).to(device)
        for i in range(z.shape[1]):
            sos_label[i][0] = 1

        sos_box_vecs = self.layout_embd.get_box_embedding(sos_box.unsqueeze(0))
        sos_label_vecs = self.layout_embd.group_label_embed(sos_label.unsqueeze(0))
        sos_tgt = self.proj_cat_tgt(torch.cat((sos_label_vecs, sos_box_vecs), -1)).squeeze(0)
        tgt[0] = sos_tgt

        out = torch.zeros(max_group_num, z.shape[1], self.cfg.d_model).to(device)  # (G,N,E)

        G = tgt.size(0)
        for i in range(G):
            o = self.decoder(tgt[:i+1], z)
            out[i] = o[i]
            rec_box_i = self.box_fcn(out[i])
            rec_label_i = self.label_fcn(out[i])
            rec_bboxes[i] = rec_box_i  # (N,4*_)
            rec_labels[i] = rec_label_i  # (N,_)
            if i < (G-1):
                tgt_label_vecs = self.layout_embd.group_label_embed(rec_label_i)  # (N,128)
                tgt_box_vecs = self.layout_embd.get_box_embedding(rec_box_i.reshape(-1, 4, max(self.cfg.discrete_x_grid, self.cfg.discrete_y_grid)).argmax(-1).unsqueeze(0)).squeeze(0)  # (N,4*128)
                tgt[i+1] = self.proj_cat_tgt(torch.cat((tgt_label_vecs, tgt_box_vecs), -1))  # (N,E)

        out = out[:-2]  # remove eos group embedding
        return out, rec_bboxes, rec_labels  # (G,N,E) / (G,N,4*_) / (G,N,_)


class ElementDecoder(nn.Module):
    def __init__(self, cfg, layout_embd):
        super(ElementDecoder, self).__init__()

        self.cfg = cfg
        self.layout_embd = layout_embd
        self.proj_cat_memory = nn.Sequential(nn.Linear(cfg.d_model+cfg.d_model, cfg.d_model))

        square_subsequent_mask = utils._generate_square_subsequent_mask(cfg.max_num_elements+2)
        self.register_buffer("square_subsequent_mask", square_subsequent_mask)

        decoder_layer = transformer.TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = transformer.LayerNorm(cfg.d_model)
        self.decoder = transformer.TransformerDecoder(decoder_layer, cfg.n_layers_decoder, decoder_norm)

        self.label_fcn = nn.Sequential(nn.Linear(cfg.d_model, 128),
                                       nn.Linear(128, self.cfg.num_labels+3),
                                       nn.ReLU(inplace=True),)
        self.box_fcn = nn.Sequential(nn.Linear(cfg.d_model, 4*max(self.cfg.discrete_x_grid, self.cfg.discrete_y_grid)),
                                     nn.ReLU(inplace=True),)

    def forward(self, label, box, memory, z, mask):
        '''
        label(S,G,N,1), box(S,G,N,4*_), memory(G,N,E), mask(N,G,S), z(1,N,E)
        '''

        z = z.repeat(memory.size(0), 1, 1).unsqueeze(0)  # (1,G,N,E)
        memory = memory.unsqueeze(0)  # (1,G,N,E)
        memory, z, label, box = utils._pack_group_batch(memory, z, label, box)  # (1,G*N,E) / (1,G*N,E) / (S,G*N,1) / (S,G*N,E)
        memory = self.proj_cat_memory(torch.cat((memory, z), -1))

        N, G, S = mask.shape
        mask = utils._make_seq_first(mask).reshape(G*N, S)  # (G*N,S)
        # key_padding_mask = get_key_padding_mask(mask)

        tgt = self.layout_embd(label.squeeze(-1), box)  # (S,G*N,E)

        S = tgt.size(0)
        out = self.decoder(tgt[:S], memory, tgt_mask=self.square_subsequent_mask[:S, :S])  # , tgt_key_padding_mask=key_padding_mask)
        rec_box = self.box_fcn(out)  # (S,G*N,4*d_box)
        rec_label = self.label_fcn(out)  # (S,G*N,max_label_num+3)
        rec_box, rec_label = utils._unpack_group_batch(N, rec_box, rec_label)  # (S,G,N,4*_) / (S,G,N,_)

        return rec_box, rec_label  # (S,G,N,4*_) / (S,G,N,_)

    def inference(self, memory, z, max_num_elements, device):
        # inference
        G, N, E = memory.shape
        z = z.repeat(memory.size(0), 1, 1).unsqueeze(0)  # (1,G,N,E)
        memory = memory.unsqueeze(0)  # (1,G,N,E)
        memory, z = utils._pack_group_batch(memory, z)  # (1,G*N,E) / (1,G*N,E)
        memory = self.proj_cat_memory(torch.cat((memory, z), -1))

        tgt = torch.zeros(max_num_elements, z.shape[1], self.cfg.d_model).to(device)  # S,G*N,_

        rec_label = torch.zeros(max_num_elements, z.shape[1], self.cfg.num_labels+3).to(device)
        rec_box = torch.zeros(max_num_elements, z.shape[1], 4*max(self.cfg.discrete_x_grid, self.cfg.discrete_y_grid)).to(device)  # (S,G*N,4*_)

        sos_box = torch.zeros(z.shape[1], 4).to(device).long()
        sos_label = torch.zeros(z.shape[1]).to(device).long()
        for i in range(z.shape[1]):
            sos_label[i] = self.cfg.num_labels+1
        sos_tgt = self.layout_embd(sos_label.unsqueeze(0), sos_box.unsqueeze(0)).squeeze(0)
        tgt[0] = sos_tgt

        out = torch.zeros(max_num_elements, z.shape[1], self.cfg.d_model).to(device)  # (S,G*N,E)

        S = tgt.size(0)
        for i in range(S):
            o = self.decoder(tgt[:i+1], memory)
            out[i] = o[i]
            rec_box_i = self.box_fcn(out[i])  # (G*N,4*E)
            rec_label_i = self.label_fcn(out[i])  # (G*N,E)
            rec_box[i] = rec_box_i
            rec_label[i] = rec_label_i
            if i < (S-1):
                tgt[i+1] = self.layout_embd(rec_label_i.argmax(1).unsqueeze(0), rec_box_i.reshape(-1, 4, max(self.cfg.discrete_x_grid, self.cfg.discrete_y_grid)).argmax(-1).unsqueeze(0)).squeeze(0)

        rec_box, rec_label = utils._unpack_group_batch(N, rec_box, rec_label)  # (S,G,N,4*_) / (S,G,N,_)
        return rec_box, rec_label


class C2FLayoutTransformer(nn.Module):
    def __init__(self, cfg):
        super(C2FLayoutTransformer, self).__init__()
        self.cfg = cfg
        self.layout_embd = LayoutEmbedding(cfg)
        self.encoder = Encoder(cfg, self.layout_embd)
        self.vae = VAE(cfg)
        self.group_decoder = GroupDecoder(cfg, self.layout_embd)
        self.ele_decoder = ElementDecoder(cfg, self.layout_embd)

    def forward(self, data, device, feed_gt=True):

        masks = data['masks']  # (N,S)
        group_masks = data['group_masks']  # (N,G)
        grouped_ele_masks = data['grouped_ele_masks']  # (N,G,S)

        # make sequence dimention first
        #   box
        grouped_box = data['grouped_bboxes']
        N, G, S, E = grouped_box.shape
        grouped_box = utils._make_seq_first(grouped_box.reshape(N, G*S, E))  # (G*S,N,4)
        bboxes, group_bounding_box, grouped_box = utils._make_seq_first(
            data['bboxes'],  # (S,N,4)
            data['group_bounding_box'],  # (G,N,4)
            grouped_box.reshape(G, S, N*E)  # (S,G,N*4)
        )
        grouped_box = grouped_box.reshape(S, G, N, E)
        #   label
        labels = utils._make_seq_first(data['labels'].unsqueeze(2)).squeeze(2)  # (S,N)
        label_in_one_group = utils._make_seq_first(data['label_in_one_group'])  # (G,N,_)
        grouped_label = utils._make_seq_first(data['grouped_labels'].reshape(N, G*S, 1))  # (G*S,N,1)
        grouped_label = utils._make_seq_first(grouped_label.reshape(G, S, N)).unsqueeze(3)  # (S,G,N,1)

        # feed to transformer
        memory = self.encoder(labels, bboxes, masks)

        z, mu, logvar = self.vae(memory)

        if feed_gt:
            # Region decoder
            group_embd, rec_group_bounding_box, rec_label_in_one_group = self.group_decoder(label_in_one_group, group_bounding_box, z, group_masks)  # (G,N,E) / (G+2,N,4*_) / (G+2,N,_)
            # Element decoder
            rec_box, rec_label = self.ele_decoder(grouped_label, grouped_box, group_embd, z, grouped_ele_masks)  # (S,G,N,4*_) / (S,G,N,_)
        else:
            # Region decoder
            max_group_num = label_in_one_group.shape[0]
            group_embd, rec_group_bounding_box, rec_label_in_one_group = self.group_decoder.inference(z, device, max_group_num)  # (G,N,E) / (G+2,N,4*_) / (G+2,N,_)
            # Element decoder
            max_num_elements = grouped_label.shape[0]
            rec_box, rec_label = self.ele_decoder.inference(group_embd, z, max_num_elements, device)  # (S,G,N,4*_) / (S,G,N,_)

        rec_box, rec_label = utils._make_group_first(rec_box, rec_label)  # (G,S,N,4*_) / (G,S,N,_)

        rec_box, rec_label, rec_group_bounding_box, rec_label_in_one_group, mu, logvar, z = utils._make_batch_first(
            rec_box.reshape(G*S, N, -1),      # --> (N,G*S,4*_)
            rec_label.reshape(G*S, N, -1),    # --> (N,G*S,_)
            rec_group_bounding_box,         # --> (N,G+2,4*_)
            rec_label_in_one_group,         # --> (N,G+2,_)
            mu, logvar, z                   # --> (N,1,E)
        )
        rec_box = rec_box.reshape(N, G, S, -1)  # (N,G,S,4*_)
        rec_label = rec_label.reshape(N, G, S, -1)  # (N,G,S,_)

        rec_box = rec_box.reshape(N, G, S, 4, -1)
        rec_group_bounding_box = rec_group_bounding_box.reshape(N, G+2, 4, -1)

        ori = {
            'bboxes': data['bboxes'],
            'labels': data['labels'],
            'masks': data['masks'],
            'group_bounding_box': data['group_bounding_box'],
            'label_in_one_group': data['label_in_one_group'],
            'group_masks': data['group_masks'],
            'grouped_bboxes': data['grouped_bboxes'],
            'grouped_labels': data['grouped_labels'],
            'grouped_ele_masks': data['grouped_ele_masks']
            }
        rec = {
            'group_bounding_box': rec_group_bounding_box,    # (N,G+2,4,_)
            'label_in_one_group': rec_label_in_one_group,    # (N,G+2,_)
            'grouped_bboxes': rec_box,                          # (N,G,S,4,_)
            'grouped_labels': rec_label                       # (N,G,S,_)
        }
        kl_info = {
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

        return ori, rec, kl_info

    def inference(self, device, z=None):
        z = self.vae.inference(z, device)

        # Region decoder
        max_group_num = self.cfg.max_num_elements+2
        group_embd, gen_group_bounding_box, gen_label_in_one_group = self.group_decoder.inference(z, device, max_group_num)  # (G,N,E) / (G+2,N,4*_) / (G+2,N,_)
        # Element decoder
        max_num_elements = self.cfg.max_num_elements+2
        gen_box, gen_label = self.ele_decoder.inference(group_embd, z, max_num_elements, device)  # (S,G,N,4*_) / (S,G,N,_)

        gen_box, gen_label = utils._make_group_first(gen_box, gen_label)  # (G,S,N,4*_) / (G,S,N,_)

        G, S, N, _ = gen_box.shape
        gen_box, gen_label, gen_group_bounding_box, gen_label_in_one_group, = utils._make_batch_first(
            gen_box.reshape(G*S, N, -1),      # --> (N,G*S,4*_)
            gen_label.reshape(G*S, N, -1),    # --> (N,G*S,_)
            gen_group_bounding_box,         # --> (N,G+2,4*_)
            gen_label_in_one_group,         # --> (N,G+2,_)
        )
        gen_box = gen_box.reshape(N, G, S, -1)  # (N,G,S,4*_)
        gen_label = gen_label.reshape(N, G, S, -1)  # (N,G,S,_)

        gen_box = gen_box.reshape(N, G, S, 4, -1)
        gen_group_bounding_box = gen_group_bounding_box.reshape(N, G+2, 4, -1)

        gen = {
            'group_bounding_box': gen_group_bounding_box,    # (N,G+2,4,_)
            'label_in_one_group': gen_label_in_one_group,    # (N,G+2,_)
            'grouped_bboxes': gen_box,                          # (N,G,S,4,_)
            'grouped_labels': gen_label,                      # (N,G,S,_)
        }

        return gen
