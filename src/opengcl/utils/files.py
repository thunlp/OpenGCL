import urllib
import ssl
import os
import os.path as osp
import errno

ssl._create_default_https_context = ssl._create_unverified_context


# methods from torch_geometrics
def files_exist(files):
    return all([osp.exists(f) for f in files])


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('[downloader] Using exist file', filename)
        return path

    if log:
        print('[downloader] Downloading', url)

    makedirs(folder)
    data = urllib.request.urlopen(url)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path
