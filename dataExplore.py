
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import os

FL, FW = 105.0, 68.0  # meters

def sb_to_spadl_xy(x, y, fidelity_version=None, assume_cell_center=False):
    """
    Convert a single StatsBomb (x,y) to SPADL meters.
    - If `assume_cell_center=True`, subtract half-cell. Use for old, integer-ish event coords.
    - For 360 freeze-frames, pass assume_cell_center=False (no center shift).
    """
    if assume_cell_center:
        cell_side = 0.1 if fidelity_version == 2 else 1.0
        x = x - cell_side/2.0
        y = y - cell_side/2.0

    x_m = np.clip(x / 120.0 * FL, 0, FL)
    y_m = np.clip(FW - (y / 80.0 * FW), 0, FW)
    return x_m, y_m

def ltr_flip_if_away(x_m, y_m, is_away):
    if is_away:
        return (FL - x_m, FW - y_m)
    return (x_m, y_m)

# Example: transform a 360 freeze-frame list `ff` for the same LTR convention as SPADL
def transform_freeze_frame(ff, is_away, fidelity_version=None):
    out = []
    for p in ff:
        x, y = p["location"]
        xm, ym = sb_to_spadl_xy(x, y, fidelity_version, assume_cell_center=True)  # key difference
        xm, ym = ltr_flip_if_away(xm, ym, is_away)
        q = dict(p)
        q["location_spadl"] = [xm, ym]
        out.append(q)
    return out

def save_ids(api):
    count = 0
    game_ids = []
    for entry in os.scandir("open-data-master/data/three-sixty"):
        if entry.is_file():
            id = int(entry.name.split('.')[0])
            try:
                events = api.events(game_id = id, load_360 = True)
                game_ids.append(id)
            except:
                count += 1
                continue


    print(f'Example Game IDs: {game_ids[3:6]}')
    print(f'Total Games: {len(game_ids)}')
    print(f"Faulty Game Count: {count}/{len(game_ids)}")

    np.save('game_ids.npy',np.array(game_ids))


if __name__ == '__main__':
    api = StatsBombLoader(getter="local", root="open-data-master/data")
    game_ids = np.load('game_ids.npy')    
    fucked_count = 0
    for id in game_ids:
        events = api.events(game_id = id, load_360 = True)
        fidelity = spadl.statsbomb._infer_xy_fidelity_versions(events)
        team_id = events[0:1].team_id.item()

        spadl_actions = spadl.statsbomb.convert_to_actions(events,home_team_id=team_id)
        spadl_actions = spadl.add_names(spadl_actions) # add actiontype and result names
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=team_id)
        
        spadl_passes = spadl_actions_l2r[(spadl_actions_l2r.type_id == 0).tolist()]
        
        frame_ids = spadl_passes.original_event_id.tolist()

        three_sixty = events.freeze_frame_360[events['event_id'].isin(frame_ids)].tolist()
        transformed_threesixty = transform_freeze_frame(three_sixty,is_away=False,fidelity_version=fidelity) # need to be able to handle batch of all data

        if len(frame_ids) != len(three_sixty):
            fucked_count += 1
            print(f"Fucked{fucked_count}")


        break

    
        
