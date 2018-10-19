import numpy as np
import menpo.io as mio
import scipy.io as sio

from pathlib import Path
from menpo.visualize import print_dynamic, print_progress
from menpo.shape import PointCloud
from menpo.image import Image

from ... import utils


def mpii_iterator(is_training, base=384, full_joint=False):
    database_path = Path('/vol/atlas/databases/body/MPIIHumanPose')

    annotations = sio.loadmat(str(
        database_path / 'mpii_human_pose_v1_u12_1.mat'), squeeze_me=True, struct_as_record=False)
    annotations = annotations['RELEASE']

    pts_index = np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8, 9],
                          [2, 6, 3],
                          [10, 11, 12],
                          [12, 7, 13],
                          [13, 14, 15]])

    for anno in print_progress(
            annotations.annolist[annotations.img_train == is_training]):

        img_name = Path(anno.image.name)

        try:
            img = mio.import_image(
                '{}/images/{}'.format(database_path, img_name), normalise=False)
        except Exception as e:
            print('image {} not found'.format(img_name))
            print(e)
            continue

        rects = anno.annorect
        if not type(rects) == np.ndarray:
            rects = np.array([rects])

        for j, rect in enumerate(rects):
            pimg = img.copy()

            try:
                anno_pts = rect.annopoints
            except:
                print('no annotation found for image {}-rect {} '.format(
                    img_name, j))
                continue

            if type(anno_pts) == np.ndarray:
                joints = anno_pts
            else:
                joints = anno_pts.point

            joints_lms = np.zeros((16, 2)) - 10000
            visiblepts = [8, 9]
            marked_ids = []

            if not type(joints) == np.ndarray:
                continue

            for joint in joints:
                x = joint.x
                y = joint.y
                joints_lms[joint.id] = [y, x]
                marked_ids.append(joint.id)

                if type(joint.is_visible) == list:
                    if len(joint.is_visible) > 0:
                        print('unknown situation')
                else:
                    visiblepts.append(joint.id)

            if full_joint and len(marked_ids) != 16:
                continue

            if not (joints_lms == np.zeros((16, 2)) - 10000).all():

                scale = 0
                torsor = []
                if 2 in visiblepts and 13 in visiblepts:
                    torsor.append(joints_lms[2])
                    torsor.append(joints_lms[13])
                    scale = np.max([scale, np.linalg.norm(
                        joints_lms[13] - joints_lms[2])])

                if 12 in visiblepts and 3 in visiblepts:
                    torsor.append(joints_lms[12])
                    torsor.append(joints_lms[3])
                    scale = np.max([scale, np.linalg.norm(
                        joints_lms[12] - joints_lms[3])])

                if len(torsor) == 0:
                    print('torsor not found')
                    continue

                centre = PointCloud(torsor).centre_of_bounds()
                scale /= 60.

                pimg.landmarks['JOINT'] = PointCloud(joints_lms)
                pimg, trans, c_scale = utils.crop_image(
                    pimg, centre, scale, [384, 384], base=base)

                yield {
                    'image': pimg,
                    'visible_pts': visiblepts,
                    'marked_index': marked_ids
                }


def lsp_iterator(is_training, base=384, full_joint=False):
    annotations = sio.loadmat(
        '/vol/atlas/databases/body/lsp_dataset/joints.mat', squeeze_me=True, struct_as_record=False)
    image_load_path = Path('/vol/atlas/databases/body/lsp_dataset/images')

    pts_index = np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8, 9],
                          [2, 6, 3],
                          [10, 11, 12],
                          [12, 7, 13],
                          [13, 14, 15]])

    indexes = range(1, 1001) if is_training else range(1001, 2001)

    for nimg in print_progress(list(indexes)):

        image_name = Path('im{:04d}.jpg'.format(nimg))
        load_path = image_load_path / image_name

        pimg = mio.import_image(load_path, normalize=False)

        anno_points = annotations['joints'][:2, :, nimg - 1].T[:, -1::-1]
        joints_lms = np.zeros((16, 2)) - 10000
        visiblepts = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]
        marked_ids = visiblepts
        joints_lms[marked_ids] = anno_points
        joints_lms[6] = (joints_lms[2] + joints_lms[3]) / 2.
        joints_lms[7] = (joints_lms[12] + joints_lms[13]) / 2.
        visiblepts += [6, 7]
        marked_ids += [6, 7]

        if full_joint and len(marked_ids) != 16:
            continue

        pimg.landmarks['JOINT'] = PointCloud(joints_lms)

        scale = 0
        torsor = []
        if 2 in visiblepts and 13 in visiblepts:
            torsor.append(joints_lms[2])
            torsor.append(joints_lms[13])
            scale = np.max([scale, np.linalg.norm(
                joints_lms[13] - joints_lms[2])])

        if 12 in visiblepts and 3 in visiblepts:
            torsor.append(joints_lms[12])
            torsor.append(joints_lms[3])
            scale = np.max([scale, np.linalg.norm(
                joints_lms[12] - joints_lms[3])])

        if len(torsor) == 0:
            print('torsor not found')
            continue

        centre = PointCloud(torsor).centre_of_bounds()
        scale /= 60.

        pimg.landmarks['JOINT'] = PointCloud(joints_lms)

        pimg, trans, c_scale = utils.crop_image(
            pimg, centre, scale, [384, 384], base=base)

        yield {
            'image': pimg,
            'visible_pts': visiblepts,
            'marked_index': marked_ids
        }


def lsp_extended(base=384, full_joint=False):
    annotations = sio.loadmat(
        '/vol/atlas/databases/body/lspet_dataset_extend/joints.mat', squeeze_me=True, struct_as_record=False)
    image_load_path = Path(
        '/vol/atlas/databases/body/lspet_dataset_extend/images')

    pts_index = np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8, 9],
                          [2, 6, 3],
                          [10, 11, 12],
                          [12, 7, 13],
                          [13, 14, 15]])

    indexes = range(1, 10001)

    for nimg in print_progress(list(indexes)):

        image_name = Path('im{:05d}.jpg'.format(nimg))
        load_path = image_load_path / image_name

        pimg = mio.import_image(load_path, normalize=False)

        anno_points = annotations['joints'][:, -2::-1, nimg - 1]
        joints_lms = np.zeros((16, 2)) - 10000
        marked_ids = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9])
        visiblepts = marked_ids[(anno_points > 0).all(axis=1)]

        joints_lms[marked_ids] = anno_points
        joints_lms[6] = (joints_lms[2] + joints_lms[3]) / 2.
        joints_lms[7] = (joints_lms[12] + joints_lms[13]) / 2.
        marked_ids = list(marked_ids) + [6, 7]
        visiblepts = list(visiblepts)

        if full_joint and len(marked_ids) != 16:
            continue

        if 2 in visiblepts and 3 in visiblepts:
            visiblepts.append(6)

        if 12 in visiblepts and 13 in visiblepts:
            visiblepts.append(7)

        pimg.landmarks['JOINT'] = PointCloud(joints_lms)

        scale = 0
        torsor = []
        if 2 in visiblepts and 13 in visiblepts:
            torsor.append(joints_lms[2])
            torsor.append(joints_lms[13])
            scale = np.max([scale, np.linalg.norm(
                joints_lms[13] - joints_lms[2])])

        if 12 in visiblepts and 3 in visiblepts:
            torsor.append(joints_lms[12])
            torsor.append(joints_lms[3])
            scale = np.max([scale, np.linalg.norm(
                joints_lms[12] - joints_lms[3])])

        if len(torsor) == 0:
            print('torsor not found')
            continue

        centre = PointCloud(torsor).centre_of_bounds()
        scale /= 60.

        pimg.landmarks['JOINT'] = PointCloud(joints_lms)

        pimg, trans, c_scale = utils.crop_image(
            pimg, centre, scale, [384, 384], base=base)

        yield {
            'image': pimg,
            'visible_pts': visiblepts,
            'marked_index': marked_ids
        }


def posetrack_iterator(istraining=1, base=384):
    folder = 'train' if istraining else 'val'
    db_path = Path('/vol/atlas/databases/body/PoseTrack/posetrack_data/')
    annotation_path = db_path / 'annotations' / folder

    for anno_path in print_progress(list(annotation_path.glob('*.mat'))):

        annotations = sio.loadmat(
            str(anno_path), squeeze_me=True, struct_as_record=False)['annolist']

        for anno in annotations:
            if not anno.is_labeled:
                continue

            image_path = db_path / anno.image.name
            img = mio.import_image(image_path, normalize=False)

            if not type(anno.annorect) == np.ndarray:
                anno.annorect = np.array([anno.annorect])

            for rect in anno.annorect:
                pimg = img.copy()
                joints_lms = np.zeros((16, 2)) - 10000

                marked_ids = []
                visiblepts = []
                pts_idx = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, -1, 9]
                points = None
                try:
                    if type(rect.annopoints) == np.ndarray:
                        points = rect.annopoints
                    else:
                        points = rect.annopoints.point
                except:
                    continue

                if not type(points) == np.ndarray:
                    points = np.array([points])

                for p in points:
                    pid = pts_idx[p.id]
                    if pid < 0:
                        continue

                    joints_lms[pid] = np.array([p.y, p.x])

                    marked_ids.append(pid)
                    if p.is_visible:
                        visiblepts.append(pid)

                joints_lms[6] = (joints_lms[2] + joints_lms[3]) / 2.
                joints_lms[7] = (joints_lms[12] + joints_lms[13]) / 2.

                marked_ids = list(marked_ids) + [6, 7]
                visiblepts = list(visiblepts)

                if 2 in visiblepts and 3 in visiblepts:
                    visiblepts.append(6)

                if 12 in visiblepts and 13 in visiblepts:
                    visiblepts.append(7)

                pimg.landmarks['JOINT'] = PointCloud(joints_lms)

                scale = 0
                torsor = []
                if 2 in visiblepts and 13 in visiblepts:
                    torsor.append(joints_lms[2])
                    torsor.append(joints_lms[13])
                    scale = np.max([scale, np.linalg.norm(
                        joints_lms[13] - joints_lms[2])])

                if 12 in visiblepts and 3 in visiblepts:
                    torsor.append(joints_lms[12])
                    torsor.append(joints_lms[3])
                    scale = np.max([scale, np.linalg.norm(
                        joints_lms[12] - joints_lms[3])])

                if len(torsor) == 0:
                    print('torsor not found')
                    continue

                centre = PointCloud(torsor).centre_of_bounds()
                scale /= 60.

                pimg, trans, c_scale = utils.crop_image(
                    pimg, centre, scale, [384, 384], base=base)

                yield {
                    'image': pimg,
                    'visible_pts': visiblepts,
                    'marked_index': marked_ids
                }


def densereg_face_iterator(istraining, db='helen'):

    database_path = Path(
        '/vol/atlas/homes/yz4009/databases/DenseReg/FaceRegDataset') / db
    landmarks_path = Path('/vol/atlas/databases') / db

    image_folder = 'Images_Resized'
    uv_folder = 'Labels_Resized'
    z_folder = 'Labels_Resized'

    if istraining == 1:
        database_path = database_path / 'trainset'
        landmarks_path = landmarks_path / 'trainset'
    elif istraining == 0:
        database_path = database_path / 'testset'
        landmarks_path = landmarks_path / 'testset'


    for pimg in print_progress(mio.import_images(database_path / image_folder / image_folder)):

        image_name = pimg.path.stem

        # load iuv data
        labels = sio.loadmat(
            str(database_path / uv_folder / uv_folder / ('%s.mat' % image_name)))
        iuv = Image(np.stack([
            (labels['LabelsH_resized'] >= 0).astype(np.float32),
            labels['LabelsH_resized'],
            labels['LabelsV_resized']
        ]).clip(0, 1))

        # load lms data
        try:
            orig_image = mio.import_image(
                landmarks_path / ('%s.jpg' % image_name))
        except:
            orig_image = mio.import_image(
                landmarks_path / ('%s.png' % image_name))

        orig_image = orig_image.resize(pimg.shape)

        pimg.landmarks['JOINT'] = orig_image.landmarks['PTS']

        pimg_data = utils.crop_image_bounding_box(
            pimg,
            pimg.landmarks['JOINT'].bounding_box(),
            [384, 384],
            base=256)
        
        pimg = pimg_data[0]

        iuv_data = utils.crop_image_bounding_box(
            iuv,
            pimg.landmarks['JOINT'].bounding_box(),
            [384, 384],
            base=256)

        iuv = iuv_data[0]
        
        yield {
            'image': pimg,
            'iuv': iuv,
            'visible_pts': np.array(list(range(68))),
            'marked_index': np.array(list(range(68)))
        }


def densereg_pose_iterator():
    database_path = Path(
        '/vol/atlas/homes/yz4009/databases/DenseReg/IBUG_UP_DATA/')
    image_path = database_path / 'ProcessImagesStatic'
    iuv_path = database_path / 'Processed_correspondence_Static'
    
    iuv_map = sio.loadmat(str(database_path/'GMDS.mat'))
    index_i, index_u, index_v = iuv_map['Index'].squeeze(),iuv_map['Final_U'].squeeze(),iuv_map['Final_V'].squeeze()
    index_i = np.concatenate([[0], index_i])
    index_u = np.concatenate([[0], index_u])
    index_v = np.concatenate([[0], index_v])

    for pimg in print_progress(mio.import_images(image_path)[1:]):

        image_name = pimg.path.stem
        bbox = PointCloud(pimg.bounds()).bounding_box()
        # load iuv data
        iuv_mat = sio.loadmat(
            str(iuv_path / ('%s.mat' % image_name)))['I_correspondence']
        iuv = Image(np.stack([index_i[iuv_mat], index_u[iuv_mat] * 255, index_v[iuv_mat]*255]).astype(np.uint8))

        # load lms data

        pimg_data = utils.crop_image_bounding_box(
            pimg,
            bbox,
            [384, 384],
            base=256)
        
        pimg = pimg_data[0]

        iuv_data = utils.crop_image_bounding_box(
            iuv,
            bbox,
            [384, 384],
            base=256,
            order=0)
        
        iuv = iuv_data[0]
        
        yield {
            'image': pimg,
            'iuv': iuv
        }
