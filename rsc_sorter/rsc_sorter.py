from os import walk
import os
import re

UNSORTED_RSC_PATH = 'rsc_unsorted'
SORTED_RSC_PATH = 'rsc'

def sort_rsc():
    UNSORTED_RSC_PATH = 'rsc_unsorted'
    SORTED_RSC_PATH = 'rsc'

    _, _, filenames = next(walk(UNSORTED_RSC_PATH))

    # %%
    maps = []
    reps = []
    for f in filenames:
        ext = f.split(".")[-1]
        if ext == 'osr':
            reps.append(f)
        elif ext == 'osu':
            maps.append(f)

    # %%
    for map in maps:
        map_no_ext = map[:-4]
        if len(reps) == 0:
            print("No Maps to Sort.")
            break
        dir     = f"{SORTED_RSC_PATH}/{map_no_ext}"
        dir_rep = f"{SORTED_RSC_PATH}/{map_no_ext}/rep"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(dir_rep):
            os.makedirs(dir_rep)

        os.replace(f'{UNSORTED_RSC_PATH}/{map}',
                   f'{dir}/{map}')

        print(f'Map From: {UNSORTED_RSC_PATH}/{map}')
        print(f'Map To:   {dir}/{map}')

        matcher = re.sub(r"\([^()]+\)(?=[^()]*)\s(\[.*\])", r'\1', map_no_ext)
        print('Sorting For: ', matcher)
        for r in reps:
            if r.count(matcher):
                print(f'Rep From: {UNSORTED_RSC_PATH}/{r}',
                      f'Rep To:   {dir_rep}/{r}')
                os.replace(f'{UNSORTED_RSC_PATH}/{r}',
                           f'{dir_rep}/{r}')


