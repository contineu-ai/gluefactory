 
import argparse
import os
import requests
import subprocess

from typing import List, Tuple
from tqdm import tqdm

# https://zenodo.org/api/records/10462003/files/berlinStreet.tar.zst.ab/content

#---------------------------------------------------------------
# The Zenodo base url.
#---------------------------------------------------------------
base_url = "https://zenodo.org"

#---------------------------------------------------------------
# Scenes list.
#---------------------------------------------------------------
scenes = ['bank', 'barbershop', 'berlin', 'classroom', 'garage', 'harmony', 'italianFlat', 'kartu', 'loneMonk',
          'medievalPort', 'middleEast', 'passion', 'rainbow', 'seoul', 'shapespark', 'showroom', 'simple', 'tokyo',
          'urbanCanyon', 'vitoria', 'warehouse', 'berlinStreet', 'church', 'corridors', 'meetingRoom1', 'meetingRoom2',
          'stadium', 'townSquare', 'trainStation', 'uni']

#---------------------------------------------------------------
# Record-Number to Record-ID mapping.
#---------------------------------------------------------------
record_id_map = {1: '10459385',
                 2: '10446255',
                 3: '10446267',
                 4: '10459822',
                 5: '10461380',
                 6: '10461536',
                 7: '10461725',
                 8: '10461814',
                 9: '10462003'}

#---------------------------------------------------------------
# Scene-record mapping.
#---------------------------------------------------------------
scene_record_map = {'bank': [(1, 'bank.tar.zst.aa'), (8, 'bank.tar.zst.ab')], #✅🔥  -- doubt
                    'barbershop': [(9, 'barbershop.tar.zst.aa')], #✅🔥
                    'berlin': [(8, 'berlin.tar.zst.aa')], #✅🔥
                    'classroom': [(7, 'classroom.tar.zst.aa')], #✅🔥
                    'garage': [(1, 'garage.tar.zst.aa'), (1, 'garage.tar.zst.ab'), (9, 'garage.tar.zst.ac')], #✅🔥
                    'harmony': [(7, 'harmony.tar.zst.aa')], #✅🔥
                    'italianFlat': [(8, 'italianFlat.tar.zst.aa')], #✅🔥
                    'kartu': [(6, 'kartu.tar.zst.aa')], #✅🔥
                    'loneMonk': [(1, 'loneMonk.tar.zst.aa'), (8, 'loneMonk.tar.zst.ab')], #✅🔥
                    'medievalPort': [(1, 'medievalPort.tar.zst.aa'), (2, 'medievalPort.tar.zst.ab'), #✅🔥
                                    (2, 'medievalPort.tar.zst.ac'), (8, 'medievalPort.tar.zst.ad')],
                    'middleEast': [(2, 'middleEast.tar.zst.aa'), (2, 'middleEast.tar.zst.ab'), #✅🔥
                                    (2, 'middleEast.tar.zst.ac'), (3, 'middleEast.tar.zst.ad'), 
                                    (3, 'middleEast.tar.zst.ae'), (3, 'middleEast.tar.zst.af'),
                                    (7, 'middleEast.tar.zst.ag')],
                    'passion': [(7, 'passion.tar.zst.aa')], #✅🔥
                    'rainbow': [(3, 'rainbow.tar.zst.aa'), (8, 'rainbow.tar.zst.ab')], #✅🔥
                    'seoul': [(8, 'seoul.tar.zst.aa')], #✅🔥
                    'shapespark': [(3, 'shapespark.tar.zst.aa'), (8, 'shapespark.tar.zst.ab')], #✅🔥
                    'showroom': [(4, 'showroom.tar.zst.aa'), (4, 'showroom.tar.zst.ab'), (9, 'showroom.tar.zst.ac')], #✅🔥
                    'simple': [(7, 'simple.tar.zst.aa')], #✅🔥
                    'tokyo': [(8, 'tokyo.tar.zst.aa')], #✅🔥
                    'urbanCanyon': [(4, 'urbanCanyon.tar.zst.aa'), (4, 'urbanCanyon.tar.zst.ab'), #✅🔥
                                    (4, 'urbanCanyon.tar.zst.ac'), (5, 'urbanCanyon.tar.zst.ad'),
                                    (5, 'urbanCanyon.tar.zst.ae'), (5, 'urbanCanyon.tar.zst.af'),
                                    (5, 'urbanCanyon.tar.zst.ag'), (5, 'urbanCanyon.tar.zst.ah'),
                                    (6, 'urbanCanyon.tar.zst.ai'), (9, 'urbanCanyon.tar.zst.aj')],
                    'vitoria': [(7, 'vitoria.tar.zst.aa')], #✅🔥
                    'warehouse': [(6, 'warehouse.tar.zst.aa'), (6, 'warehouse.tar.zst.ab'), #✅🔥
                                  (9, 'warehouse.tar.zst.ac')],
                    'berlinStreet': [(6, 'berlinStreet.tar.zst.aa'), (9, 'berlinStreet.tar.zst.ab')], #✅🔥
                    'church': [(8, 'church.tar.zst.aa')], #✅🔥
                    'corridors': [(9, 'corridors.tar.zst.aa')], #✅🔥
                    'meetingRoom1': [(9, 'meetingRoom1.tar.zst.aa')], #✅🔥
                    'meetingRoom2': [(9, 'meetingRoom2.tar.zst.aa')], #✅🔥
                    'stadium': [(8, 'stadium.tar.zst.aa')], #✅🔥
                    'townSquare': [(8, 'townSquare.tar.zst.aa')], #✅🔥
                    'trainStation': [(7, 'trainStation.tar.zst.aa')], #✅🔥
                    'uni': [(8, 'uni.tar.zst.aa')]} #✅🔥

#=======================================================================================================================

def get_scene_urls(scene:str) -> List[Tuple[str, str]]:
    records = scene_record_map[scene]
    res = []
    #searches for every file_url individually, but files are not downloaded yet 
    for record_num, fname in records:
        #search by record_id
        response = requests.get(f'{base_url}/api/records/{record_id_map[record_num]}')
        for file in response.json()['files']:
            if fname == file['key']:
                url = file['links']['self']
                #scene and file_name added, for when saving to file
                res.append((fname, url))
    return res

#-----------------------------------------------------------------------------------------------------------------------

def download_scene(urls: List[Tuple[str, str]], dest_dir: str) -> None:
    for fname, url in urls:
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        filepath = os.path.join(dest_dir, fname)
        with open(filepath, 'wb') as f, tqdm(
            desc=fname,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

#-----------------------------------------------------------------------------------------------------------------------

def unpack_scene(dest_dir:str, scene:str) -> None:
    current_dir = os.getcwd()
    command = 'cd ' + dest_dir + ' && cat ' + scene + '.tar.zst.* | tar -I zstd -xf - && cd ' + current_dir
    subprocess.run(command, shell=True)

#-----------------------------------------------------------------------------------------------------------------------

def run(args) -> None:
    dest_dir = args.dest_dir

    #Set default if not given
    chosen_scenes = args.scenes
    if not(chosen_scenes):
        chosen_scenes = scenes

    os.makedirs(dest_dir, exist_ok=True)
    for scene in chosen_scenes:
        if scene == 'seoul':
            continue
        print(f'\nDownloading {scene}...', flush=True)
        urls = get_scene_urls(scene)
        download_scene(urls, dest_dir)
        print('done.', flush=True)

    if args.download_only:
        return

    for scene in chosen_scenes:
        print(f'\nUnpacking {scene}...', flush=True)
        unpack_scene(dest_dir, scene)
        print('done.', flush=True)

#=======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SphereCraft from Zenodo.')
    parser.add_argument('-d', '--dest_dir',
                        required=True,
                        type=str,
                        help='The destination directory. By default, files will be automatically decompressed here. '\
                             'This assumes enough disk space is available. See approximate required disk space in '\
                             'the github webpage. To only download files, pass the --download_only flag.')
    parser.add_argument('-s', '--scenes',
                        required=False,
                        type=str,
                        nargs='*',
                        choices=scenes,
                        help='By default, all scenes will be downloaded. Use this argument to prevent that and '\
                             'instead specify the scene(s) of interest.')
    parser.add_argument('--download_only',
                        required=False,
                        action='store_true',
                        help='Disable decompression after download.')
    run(parser.parse_args())