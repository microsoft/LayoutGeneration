import os
import errno
import os.path as osp


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def files_exist(files):
    return all([osp.exists(f) for f in files])
