# """
# This script can generate mp4 files for DFB tracking + event data. For quality check in November 2024.
#
# It contains a lot of paths, so it only works on Jonas' computer unless you adapt the paths.
# """
#
#
# import matplotlib.animation
# import matplotlib.pyplot as plt
# import mplsoccer
# import pandas as pd
# import slugify
# import streamlit as st
#
# matplotlib.use('TkAgg')  # make plots show in new window (for animation)
#
#
# @st.cache_resource
# def get_data_2425():
#     fpath_lineup = "C:/Users/Jonas/Downloads/dfl_test_data/lineup_070aae16bb9287911db8945ea6d085e2.csv"
#     fpath_meta = "C:/Users/Jonas/Downloads/dfl_test_data/meta_070aae16bb9287911db8945ea6d085e2.csv"
#     fpath_events = "C:/Users/Jonas/Downloads/dfl_test_data/events_070aae16bb9287911db8945ea6d085e2_jonas.csv"
#     fpath_tracking = "C:/Users/Jonas/Downloads/dfl_test_data/part-00000-tid-7902463250698069148-39e09ac1-1e39-4e1b-b4df-be23808d0fc2-210-1-c000.csv"
#
#     df_lineup = pd.read_csv(fpath_lineup)
#     df_meta = pd.read_csv(fpath_meta)
#     df_events = pd.read_csv(fpath_events)
#     df_tracking = pd.read_csv(fpath_tracking)
#
#     df_tracking = df_tracking[df_tracking["team_id"] != "referee"]  # ignore referee data
#
#     df_lineup = df_lineup.replace(-9999, None)
#     df_meta = df_meta.replace(-9999, None)
#     df_events = df_events.replace(-9999, None)
#     df_tracking = df_tracking.replace(-9999, None)
#
#     return df_lineup, df_meta, df_events, df_tracking
#
#
# @st.cache_resource
# def get_data_2324(do_process=False):
#     fpath_lineup = "C:/Users/Jonas/Downloads/dfl_test_data/2324/part-00000-tid-5445675435187215969-1dd533a7-38f7-4b5b-b0f2-951fc4df5480-3511-1-c000.csv"
#     fpath_meta = "C:/Users/Jonas/Downloads/dfl_test_data/2324/part-00000-tid-3818518431991413984-0a2b260d-43be-4e3a-b1d3-fd9b4de879ad-3505-1-c000.csv"
#     fpath_events = "C:/Users/Jonas/Downloads/dfl_test_data/2324/part-00000-tid-2431206198544713596-ee4526a9-d813-44e1-b8fd-45b023804409-3517-1-c000.csv"
#     fpath_tracking = "C:/Users/Jonas/Downloads/dfl_test_data/2324/part-00000-tid-1287162117350619020-2e8c9d90-8d21-4866-bea5-3c909c4c6b7b-4077-1-c000.csv"
#
#     df_lineup = pd.read_csv(fpath_lineup)
#     df_meta = pd.read_csv(fpath_meta)
#     df_events = pd.read_csv(fpath_events)
#
#     def get_number_of_lines(file_path):
#         with open(file_path) as f:
#             return sum(1 for _ in f)
#
#     n_lines_tracking = get_number_of_lines(fpath_tracking)
#
#     def append_to_csv(df, fpath, key_cols, overwrite_key_cols=True):
#         assert overwrite_key_cols
#
#         def assert_no_duplicate_columns(_df):
#             duplicate_columns = _df.columns[_df.columns.duplicated()]
#             assert len(duplicate_columns) == 0, f"Duplicate columns: {duplicate_columns}"
#
#         def assert_no_duplicate_keys(_df, _key_cols):
#             duplicate_keys = _df.duplicated(_key_cols)
#             assert not duplicate_keys.any(), f"Duplicate keys: {_df[duplicate_keys]}"
#
#         assert_no_duplicate_keys(df, key_cols)
#         assert_no_duplicate_columns(df)
#
#         try:
#             df_existing = pd.read_csv(fpath)
#         except FileNotFoundError:
#             df_existing = pd.DataFrame(columns=df.columns)
#
#         assert_no_duplicate_keys(df_existing, key_cols)
#         assert_no_duplicate_columns(df_existing)
#
#         if overwrite_key_cols:
#             df_existing = df_existing[~df_existing[key_cols].apply(tuple, axis=1).isin(df[key_cols].apply(tuple, axis=1))]
#
#             df_combined = pd.concat([df_existing, df], axis=0)
#
#             assert_no_duplicate_columns(df_combined)
#             assert_no_duplicate_keys(df_combined, key_cols)
#             assert_no_duplicate_columns(df)
#             assert_no_duplicate_keys(df, key_cols)
#
#             df_combined.to_csv(fpath, index=False)
#
#     def process(df_chunk):
#         for match_id, df_match in df_chunk.groupby("match_id"):
#             fpath_match = f"C:/Users/Jonas/Downloads/dfl_test_data/2324/tracking/{match_id}.csv"
#             df_match = df_match.dropna(subset=["frame", "player_id"])
#             append_to_csv(df_match, fpath_match, ["match_id", "frame", "player_id"], overwrite_key_cols=True)
#
#     # chunksize = (10 ** 6) / 2
#     # chunk_nr = 0
#     # with pd.read_csv(fpath_tracking, chunksize=chunksize) as reader:
#     #     for df_chunk in tqdm.tqdm(reader, total=n_lines_tracking // chunksize, desc="Reading df_tracking in chunks"):
#     #         chunk_nr += 1
#     #         if chunk_nr <= 43:
#     #             continue
#     #         process(df_chunk)
#     #         del df_chunk
#     #         gc.collect()
#
#     # match_id = "2db1e98618cab2f796abb1b80753c313"
#     match_id = "0d40920bf7389eac3aedadb9d37da74c"
#     # match_id = "a329013ce7ea331fac9938e307375bba"
#     # match_id = "df7c4b8b6d8da829f17bf5b8046d93bc"
#     df_lineup_match = df_lineup[df_lineup["match_id"] == match_id]
#     df_meta_match = df_meta[df_meta["match_id"] == match_id]
#     df_events_match = df_events[df_events["match_id"] == match_id]
#     df_tracking_match = pd.read_csv(f"C:/Users/Jonas/Downloads/dfl_test_data/2324/tracking/{match_id}.csv")
#
#     df_lineup_match = df_lineup_match.replace(-9999, None)
#     df_meta_match = df_meta_match.replace(-9999, None)
#     df_events_match = df_events_match.replace(-9999, None)
#     df_tracking_match = df_tracking_match.replace(-9999, None)
#
#     return df_lineup_match, df_meta_match, df_events_match, df_tracking_match
#
#
# season = st.selectbox("Test season", ["23/24", "24/25"])
# do_process = st.toggle("Partition tracking data")
# if season == "23/24":
#     df_lineup, df_meta, df_events, df_tracking = get_data_2425()
# else:
#     df_lineup, df_meta, df_events, df_tracking = get_data_2324(do_process=do_process)
#
# df_meta["match_string"] = df_meta.apply(lambda row: f"{row['competition_name']} {row['season_name']}: {row['match_day']}.ST {row['match_title']}", axis=1)
# df_meta["slugified_match_string"] = df_meta["match_string"].apply(slugify.slugify)
# slugified_match_string = df_meta["slugified_match_string"].iloc[0]
#
# teams = df_tracking["team_id"].unique().tolist()
# ball_team = "BALL"
#
# if st.toggle("Print data"):
#     st.write("df_lineup")
#     st.write(df_lineup)
#     st.write("df_meta")
#     st.write(df_meta)
#     st.write("df_events")
#     st.write(df_events)
#     st.write("df_tracking")
#     st.write(df_tracking.head(5))
#
#     st.write(df_tracking.describe())
#
# df_passes = df_events[df_events["event_type"] == "pass"]
#
# if st.toggle("Passes"):
#     st.write("df_passes")
#     st.write(df_passes)
#
#     df_passes = df_passes.sort_values(["section", "frame"])
#
#     # shuffle for testing
#     df_passes = df_passes.sample(frac=1)
#
#     for pass_nr, (_, p4ss) in enumerate(df_passes.iterrows()):
#         pitch = mplsoccer.Pitch(pitch_type="impect")
#         fig, ax = pitch.draw()
#         plt.title(f"Pass {pass_nr + 1} (Frame: {p4ss['frame']}, Section: {p4ss['section']})")
#         pitch.arrows(
#             p4ss["x_event"], p4ss["y_event"], p4ss["x_tracking_player_2"], p4ss["y_tracking_player_2"], width=2,
#             headwidth=10, headlength=10, color='blue' if p4ss["team_id_1"] == "c6ed81e07bcb710e1388ea8865eb5314" else 'red',
#             ax=ax,
#         )
#         df_tracking_frame = df_tracking[(df_tracking["frame"] == p4ss["frame"]) & (df_tracking["section"] == p4ss["section"])]
#         for team in teams:
#             df_tracking_frame_team = df_tracking_frame[df_tracking_frame["team_id"] == team]
#             if team == "BALL":
#                 pitch.scatter(
#                     x=df_tracking_frame_team["x_tracking"], y=df_tracking_frame_team["y_tracking"], color='black',
#                     ax=ax, marker="x", s=200,
#                 )
#             else:
#                 pitch.scatter(
#                     x=df_tracking_frame_team["x_tracking"], y=df_tracking_frame_team["y_tracking"],
#                     color='blue' if team == "c6ed81e07bcb710e1388ea8865eb5314" else 'red', ax=ax, s=50, marker="o",
#                 )
#
#         st.write("pass_nr", pass_nr, p4ss)
#         st.write(fig)
#
#         if pass_nr > 10:
#             break
#
# st.write("df_passes before processing", df_passes.shape)
# st.write(df_passes)
#
# with st.spinner("Indexing tracking data..."):
#     df_tracking_indexed = df_tracking.set_index(["frame", "section", "player_id"])
#
# def get_target_x_y(row):
#     receiver = row["player_id_2"]
#     try:
#         receiver_frame = df_tracking_indexed.loc[(row["frame_rec"], row["section"], receiver)]
#         return receiver_frame["x_tracking"], receiver_frame["y_tracking"]
#     except KeyError:
#         return None, None
#
# with st.spinner("Calculating target x and y..."):
#     keys = df_passes.apply(get_target_x_y, axis=1)
#
# df_passes[["target_x", "target_y"]] = pd.DataFrame(keys.tolist(), index=df_passes.index)
# df_passes = df_passes[df_passes["frame"].notna()]
#
# st.write("df_passes after processing", df_passes.shape)
# st.write(df_passes)
#
# if st.toggle("Animate", value=True):
#     available_sections = sorted(df_tracking["section"].unique().tolist())
#     print("available_sections", available_sections)
#     for section in available_sections:
#         # if section != "second_half":
#         #     continue
#         print("section", section)
#
#         df_tracking_half = df_tracking[df_tracking["section"] == section]
#         df_tracking_half = df_tracking_half.sort_values("frame")
#         df_passes_half = df_passes[df_passes["section"] == section]
#         df_home = df_tracking_half[df_tracking_half["team_id"] == teams[0]].set_index("frame")
#         df_away = df_tracking_half[df_tracking_half["team_id"] == teams[1]].set_index("frame")
#         df_ball = df_tracking_half[df_tracking_half["team_id"] == ball_team].set_index("frame")
#         if "BALL" not in df_tracking_half["team_id"].unique():
#             continue
#
#         pitch = mplsoccer.Pitch(pitch_type='impect', goal_type='line', pitch_width=68, pitch_length=105)
#         fig, ax = pitch.draw(figsize=(16 * 0.8, 10.4 * 0.8))
#         plt.title("Tracking data")
#
#         # ball, = ax.plot([], [], ms=6, markerfacecolor='w', zorder=3, **marker_kwargs)
#         # away, = ax.plot([], [], ms=10, markerfacecolor='#b94b75', **marker_kwargs)  # red/maroon
#         # home, = ax.plot([], [], ms=10, markerfacecolor='#7f63b8', **marker_kwargs)  # purple
#         ball, = ax.plot([], [], color="black", marker="x", linestyle='None', ms=20)
#         away, = ax.plot([], [], color="red", marker="o", linestyle='None', ms=10)
#         home, = ax.plot([], [], color="blue", marker="o", linestyle='None', ms=10)
#
#         pass_plot = pitch.arrows(
#             0, 0, 0, 0, width=2, headwidth=10, headlength=10, color="black",
#             # color='blue' if p4ss["team_id_1"] == "c6ed81e07bcb710e1388ea8865eb5314" else 'red',
#             ax=ax,
#         )
#         print("pass_plot", pass_plot)
#
#         frame2pass_interval = {}
#         df_tracking_half = df_tracking_half.sort_values("frame")
#         df_passes_half = df_passes_half.sort_values("frame")
#         df_passes_half = df_passes_half.set_index("frame")
#
#         # df_passes_half = df_passes_half[df_passes_half.index < 1000]
#         # df_tracking_half = df_tracking_half[df_tracking_half["frame"] < 1000]
#
#         current_passing_frame = None
#         current_passing_end_frame = None
#         next_passing_frame = df_passes_half.index[0]
#
#         all_frames = df_tracking_half.drop_duplicates("frame")["frame"]
#         for fr in all_frames:
#             if next_passing_frame is not None and fr >= next_passing_frame:
#                 # st.write("fr A", fr)
#                 current_passing_frame = next_passing_frame
#                 current_passing_end_frame = df_passes_half.loc[current_passing_frame]["frame_rec"]
#
#                 df_next = df_passes_half[df_passes_half.index > fr+1]
#                 # st.write("df_passes_half")
#                 # st.write(df_passes_half)
#                 # st.write("df_next")
#                 # st.write(df_next)
#                 next_passing_frame = df_next.index[0] if len(df_next) > 0 else None
#
#             if current_passing_frame is not None and current_passing_end_frame is not None and \
#                     fr >= current_passing_frame and fr <= current_passing_end_frame:
#                 # st.write("fr B", fr, current_passing_frame, current_passing_end_frame)
#                 frame2pass_interval[fr] = df_passes_half.loc[current_passing_frame]
#             if current_passing_end_frame is not None and fr > current_passing_end_frame:
#                 # st.write("fr C", fr)
#                 current_passing_frame = None
#                 current_passing_end_frame = None
#                 # st.write("next_passing_frame")
#                 # st.write(next_passing_frame)
#
#             # else:
#             #     frame2pass_interval[fr] = None
#
#         # st.write("frame2pass_interval")
#         # st.write(frame2pass_interval)
#
#         # for fr, p4ss in frame2pass_interval.items():
#         #     st.write("fr")
#         #     st.write(fr)
#
#         def animate(current_frame):
#             seconds = current_frame / 25
#             mmss = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
#
#             if current_frame in frame2pass_interval:
#                 pass_for_interval = frame2pass_interval[current_frame]
#                 pass_plot.set_offsets([pass_for_interval["x_event"], pass_for_interval["y_event"]])
#                 pass_plot.set_UVC(pass_for_interval["target_x"] - pass_for_interval["x_event"], pass_for_interval["target_y"] - pass_for_interval["y_event"])
#                 xpass = pass_for_interval["xpass"]
#             else:
#                 pass_plot.set_offsets([0, 0])
#                 pass_plot.set_UVC([0], [0])
#                 xpass = None
#
#             if xpass is not None:
#                 xpass = f"{xpass:.1%}"
#             ax.set_title(f"Section {section}, Frame {current_frame}, Time: {mmss}, xPass: {xpass}")
#
#             # t1 = time.time()
#             df_ball_to_plot = df_ball
#             df_home_to_plot = df_home
#             df_away_to_plot = df_away
#             # print(f"A {time.time() - t1:.2f} seconds.")
#
#             st.write("df_ball_to_plot")
#             st.write(df_ball_to_plot)
#
#             ball_frame = df_ball_to_plot.iloc[current_frame]
#             ball.set_data([ball_frame["x_tracking"]], [ball_frame["y_tracking"]])
#             # print(f"B {time.time() - t1:.2f} seconds.", ball_frame)
#             frame = ball_frame.name
#             # print(f"B {time.time() - t1:.2f} seconds.")
#
#             # print("c", df_away_to_plot)
#
#             # away.set_data(df_away_to_plot.loc[df_away["frame"] == frame, 'x_tracking'], df_away_to_plot.loc[df_away["frame"] == frame, 'y_tracking'])
#             away.set_data(df_away_to_plot.loc[frame, 'x_tracking'], df_away_to_plot.loc[frame, 'y_tracking'])
#             # print(f"C {time.time() - t1:.2f} seconds.")
#             home.set_data(df_home_to_plot.loc[frame, 'x_tracking'], df_home_to_plot.loc[frame, 'y_tracking'])
#             # print(f"Frame {i} took {time.time() - t1:.2f} seconds.")
#             return ball, away, home
#
#         # n_frames = len(df_tracking_half["frame"].unique())
#         n_frames = df_tracking_half["frame"].max()
#         # print("n_frames")
#         # print(n_frames)
#         # n_frames = len(df_ball)
#         anim = matplotlib.animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
#
#         # export mp4
#         anim.save(f'{slugified_match_string}_tracking_{section}.mp4', writer='ffmpeg', fps=25, progress_callback=lambda i, n: print(f'Saving frame {i} of {n} ({i/n:.1%}) ({section})') if i % 30 == 0 else None)
#
#         # cleanup
#         plt.close(fig)
