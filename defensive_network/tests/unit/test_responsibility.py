import importlib

import defensive_network.utility.general
import defensive_network.utility.pitch
import defensive_network.models.involvement
import defensive_network.models.responsibility

importlib.reload(defensive_network.utility.pitch)
importlib.reload(defensive_network.models.involvement)
importlib.reload(defensive_network.models.responsibility)


def test_responsibility_correctness():
    from defensive_network.tests.test_data import df_events, df_tracking

    df_involvement = defensive_network.models.involvement.get_involvement(df_events, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", model_radius=10, tracking_defender_meta_cols=["player_name", "player_position"])

    defensive_network.utility.pitch.plot_passes(df_events, df_tracking)

    dfg_responsibility = defensive_network.models.responsibility.get_responsibility_model(df_involvement, ["player_position", "unified_receiver_position", "defender_player_position"])
    df_involvement["responsibility"] = defensive_network.models.responsibility.get_responsibility(df_involvement, dfg_responsibility)
    defensive_network.utility.pitch.plot_passes_with_involvement(df_involvement, df_tracking, tracking_frame_col="frame_id", pass_frame_col="frame_id", n_passes=1000000)

    # MF -> ST
    # - MF 0.5, ST 0.5, DF 0
    # - MF 0.79, ST 0.24, DF 0
    # - MF 1.0, ST 0, ST 0
    # - MF 0, ST 0, DF 0
    # -> MF 0.5725, ST 0.148, DF 0
    #
    # ST -> ST
    # - DF 1.0, MF 0, ST 0
    #
    # DF -> MF
    # - MF 0.5, ST 1.0, ST 0.0
    #
    # ST -> MF
    # - MF 0.42, ST 0.42, DF 0
    i1 = (df_involvement["player_position"] == "MF") & (df_involvement["unified_receiver_position"] == "ST")
    assert (df_involvement.loc[i1 & (df_involvement["defender_player_position"] == "MF"), "responsibility"].round(2) == 0.57).all()
    assert (df_involvement.loc[i1 & (df_involvement["defender_player_position"] == "ST"), "responsibility"].round(2) == 0.15).all()
    assert (df_involvement.loc[i1 & (df_involvement["defender_player_position"] == "DF"), "responsibility"].round(2) == 0.0).all()
    i2 = (df_involvement["player_position"] == "ST") & (df_involvement["unified_receiver_position"] == "ST")
    assert (df_involvement.loc[i2 & (df_involvement["defender_player_position"] == "MF"), "responsibility"].round(2) == 0.0).all()
    assert (df_involvement.loc[i2 & (df_involvement["defender_player_position"] == "ST"), "responsibility"].round(2) == 0.0).all()
    assert (df_involvement.loc[i2 & (df_involvement["defender_player_position"] == "DF"), "responsibility"].round(2) == 1.0).all()
    i3 = (df_involvement["player_position"] == "DF") & (df_involvement["unified_receiver_position"] == "MF")
    assert (df_involvement.loc[i3 & (df_involvement["defender_player_position"] == "MF"), "responsibility"].round(2) == 0.5).all()
    assert (df_involvement.loc[i3 & (df_involvement["defender_player_position"] == "ST"), "responsibility"].round(2) == 0.5).all()
    assert len(df_involvement.loc[i3 & (df_involvement["defender_player_position"] == "DF"), "responsibility"]) == 0
    i4 = (df_involvement["player_position"] == "ST") & (df_involvement["unified_receiver_position"] == "MF")
    assert (df_involvement.loc[i4 & (df_involvement["defender_player_position"] == "MF"), "responsibility"].round(2) == 0.42).all()
    assert (df_involvement.loc[i4 & (df_involvement["defender_player_position"] == "ST"), "responsibility"].round(2) == 0.42).all()
    assert (df_involvement.loc[i4 & (df_involvement["defender_player_position"] == "DF"), "responsibility"].round(2) == 0.0).all()
