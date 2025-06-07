import pandas as pd
import io

test_data_str = """
frame_id,frame_id_rec,x_event,y_event,player_id_1,player_position,player_id_2,receiver_position,team_id_1,team_id_2,pass_is_successful,pass_xt,pass_is_intercepted,x_target,y_target,expected_receiver,expected_receiver_position
       0,           3,      0,      0,          a,             MF,          b,               ST,        H,        H,              True,    0.2,              False,      10,       0,                 ,                        
       3,          15,      8,      0,          b,             ST,          a,               MF,        H,        H,              True,   -0.1,              False,       0,       0,                 ,                        
      15,          40,     -5,      0,          a,             MF,          c,               ST,        H,        H,              True,    0.4,              False,      30,      10,                 ,                        
      40,          42,     30,     10,          c,             ST,          z,               DF,        H,        A,             False,   -0.2,               True,      40,       0,                b,                        ST
      42,          49,     40,      0,          z,             DF,          y,               MF,        A,        A,              True,    0.5,              False,       0,       0,                 ,                        
      49,          60,      0,      0,          y,             MF,          a,               MF,        A,        H,             False,   -0.3,               True,     -10,     -20,                x,                        ST
      60,          60,    -10,    -20,          a,             MF,           ,                 ,        H,         ,             False,   -0.1,              False,     -20,     -34,                b,                        ST
"""

df_events = pd.read_csv(io.StringIO(test_data_str.replace(" ", "")))
df_events["unified_receiver"] = df_events["player_id_2"].where(df_events["team_id_2"] == df_events["team_id_1"], df_events["expected_receiver"])
player2position = df_events.set_index("player_id_1")["player_position"].to_dict()
player2position.update(df_events.set_index("player_id_2")["receiver_position"].to_dict())
player2position.update(df_events.set_index("expected_receiver")["expected_receiver_position"].to_dict())
df_events["unified_receiver_position"] = df_events["unified_receiver"].map(player2position)
# df["event_string"] = df["player_id_1"] + " (" + df["player_position"] + ") -> " + df["unified_receiver"] + " (" + df["unified_receiver_position"] + ")"
i_success = df_events["pass_is_successful"]
i_intercepted = df_events["pass_is_intercepted"]
i_out = (~i_success) & (~i_intercepted)
df_events.loc[i_success, "event_string"] = df_events.loc[i_success].apply(lambda x: x["player_id_1"] + " (" + x["player_position"] + ") -> " + x["unified_receiver"] + " (" + x["unified_receiver_position"] + ")", axis=1)
df_events.loc[i_intercepted, "event_string"] = df_events.loc[i_intercepted].apply(lambda x: x["player_id_1"] + " (" + x["player_position"] + ") -> " + x["unified_receiver"] + " (" + x["unified_receiver_position"] + ") | Intercepted by " + x["player_id_2"] + " (" + x["receiver_position"] + ")", axis=1)
df_events.loc[i_out, "event_string"] = df_events.loc[i_out].apply(lambda x: x["player_id_1"] + " (" + x["player_position"] + ") -> " + x["unified_receiver"] + " (" + x["unified_receiver_position"] + ") | Out", axis=1)

assert df_events["unified_receiver"].notna().all()
assert df_events["unified_receiver_position"].notna().all()

test_tracking_str = """
frame_id,player_id,player_name,player_position,team_id,x_tracking,y_tracking
       0,        a,     a (MF),             MF,      H,         0,         0
       0,        b,     b (ST),             ST,      H,        12,         0
       0,        c,     c (ST),             ST,      H,        30,        10
       0,        y,     y (MF),             MF,      A,         5,         5
       0,        x,     x (ST),             ST,      A,         5,         -5
       0,        z,     z (DF),             DF,      A,         40,         0

       3,        a,     a (MF),             MF,      H,         0,         0
       3,        b,     b (ST),             ST,      H,        9,         0
       3,        c,     c (ST),             ST,      H,        30,        10
       3,        y,     y (MF),             MF,      A,         5,         5
       3,        x,     x (ST),             ST,      A,         5,         -5
       3,        z,     z (DF),             DF,      A,         40,         0

      15,        a,     a (MF),             MF,      H,         -5,         0
      15,        b,     b (ST),             ST,      H,        9,         0
      15,        c,     c (ST),             ST,      H,        30,        10
      15,        y,     y (MF),             MF,      A,         5,         5
      15,        x,     x (ST),             ST,      A,         5,         -5
      15,        z,     z (DF),             DF,      A,         40,         0

      40,        a,     a (MF),             MF,      H,         -5,         0
      40,        b,     b (ST),             ST,      H,        9,         0
      40,        c,     c (ST),             ST,      H,        30,        10
      40,        y,     y (MF),             MF,      A,         5,         5
      40,        x,     x (ST),             ST,      A,         5,         -5
      40,        z,     z (DF),             DF,      A,         40,         0

      42,        a,     a (MF),             MF,      H,         -5,         0
      42,        b,     b (ST),             ST,      H,        9,         0
      42,        c,     c (ST),             ST,      H,        30,        10
      42,        y,     y (MF),             MF,      A,         2,         2
      42,        x,     x (ST),             ST,      A,         5,         -5
      42,        z,     z (DF),             DF,      A,         40,         0

      49,        a,     a (MF),             MF,      H,         -7,         -5
      49,        b,     b (ST),             ST,      H,        9,         0
      49,        c,     c (ST),             ST,      H,        30,        10
      49,        y,     y (MF),             MF,      A,         2,         2
      49,        x,     x (ST),             ST,      A,         5,         -5
      49,        z,     z (DF),             DF,      A,         40,         0

      60,        a,     a (MF),             MF,      H,         -9,         -20
      60,        b,     b (ST),             ST,      H,        -30,         -25
      60,        c,     c (ST),             ST,      H,        30,        10
      60,        y,     y (MF),             MF,      A,         2,         2
      60,        x,     x (ST),             ST,      A,         5,         -5
      60,        z,     z (DF),             DF,      A,         40,         0
"""
df_tracking = pd.read_csv(io.StringIO(test_tracking_str.replace(" ", "")))
