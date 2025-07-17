import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

import streamlit as st

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import defensive_network.parse.drive

base_url = "https://www.soccerdonna.de"
# league_url_template = "https://www.futbin.com/24/leagues/2215/gpfbl?page={page}&version=gold%2Csilver%2Cbronze"
schedule_url = "https://www.soccerdonna.de/de/bundesliga/spielplan/wettbewerb_BL1.html"
# "https://www.soccerdonna.de/de/bundesliga/spielplan/wettbewerb_BL1.html"
headers = {
    "User-Agent": "Mozilla/5.0"
}

# def get_spielberichte_links(schedule_url):
#     res = requests.get(schedule_url, headers=headers)
#     soup = BeautifulSoup(res.text, "html.parser")
#     links = soup.find_all("a", href=True)
#
# # href="/de/sv-werder-bremen-bayer-04-leverkusen/index/spielbericht_119364.html"
#
#     player_links = []
#     for link in links:
#         href = link['href']
#         st.write(href)
#         if "spielbericht_" in href:
#             full_url = base_url + href.split('?')[0]
#             if full_url not in player_links:
#                 player_links.append(full_url)
#     return player_links


def scrape_spielbericht(url):
    is_transfermarkt_link = "transfermarkt.de" in url
    url = url.replace("index", "aufstellung")
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    all_data = []
    pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows in the DataFrame4
    pd.set_option('display.width', 1000)  # Set a wider display width
    dfs = pd.read_html(res.text, header=0)
    st.write(f"Scraping {url}", len(dfs), "tables found")

    # get date
    link = soup.find('a', href=lambda h: h and "new/datum" in h)
    date = str(link).strip().split("datum/")[-1].split("\">")[0].strip()
    date = pd.to_datetime(date, format='%Y-%m-%d', errors='coerce')

    for df in dfs:
        for _, row in df.iterrows():
            row_str = str(row)
            if is_transfermarkt_link:
                if "€" in row_str:
                    player_name = row_str.split(" (")[0].strip()

                    # remove all numbers
                    player_name = ''.join(filter(lambda x: not x.isdigit(), player_name)).replace("Unnamed: ", "").replace("NaN", "").strip()
                    market_value_str = row_str.split(" €")[0].split(", ")[-1].strip()
                    import re
                    matches = re.findall(r'\d+(?:,\d+)?', market_value_str)
                    market_value = [float(m.replace(',', '.')) for m in matches][0]
                    # market_value = int(''.join(filter(str.isdigit, market_value_str.replace(",", "."))))
                    if "Tsd." in market_value_str:
                        market_value = market_value * 1000
                    elif "Mio." in market_value_str:
                        market_value = market_value * 1000000
                    else:
                        raise ValueError("Unknown market value format: " + market_value_str)

                    player_data = {
                        'name': player_name,
                        'market_value': market_value,
                        'market_value_str': market_value_str,
                        "date": date,
                        'url': url,
                    }
                    all_data.append(player_data)
            else:
                if "MW" in row_str:
                    player_name = row_str.split(".2")[-1].split(",")[0].strip()
                    st.write("'" + player_name + "'")
                    market_value = row_str.split("MW:")[-1].split(" €")[0].strip().replace(".", "")
                    st.write(market_value, "€")
                    player_data = {
                        'name': player_name,
                        'market_value': market_value,
                        'url': url
                    }
                    all_data.append(player_data)
    df = pd.DataFrame(all_data).drop_duplicates()
    df["market_value"] = df["market_value"].astype(float, errors='ignore')
    return df


def main():
    seen_links = set()
    data = []
    spielberichte_links = [
        "https://www.soccerdonna.de/de/sv-werder-bremen-bayer-04-leverkusen/aufstellung/spielbericht_119364.html",
        "https://www.soccerdonna.de/de/rb-leipzig-sc-freiburg/index/spielbericht_119367.html",
        "https://www.soccerdonna.de/de/fc-bayern-muenchen-tsg-1899-hoffenheim/index/spielbericht_119366.html",
        "https://www.soccerdonna.de/de/eintracht-frankfurt-1-fc-koeln/index/spielbericht_119368.html",
        "https://www.soccerdonna.de/de/msv-duisburg-1-fc-nuernberg/index/spielbericht_119365.html",
        "https://www.soccerdonna.de/de/sgs-essen-vfl-wolfsburg/index/spielbericht_119369.html",
    ]
    for url in spielberichte_links:
        if url not in seen_links:
            seen_links.add(url)
            player_data = scrape_spielbericht(url)
            data.append(player_data)
            # st.write(f"Collected: {player_data['name']} ({player_data['overall']})")
            # time.sleep(1.5)
    df = pd.concat(data, ignore_index=True)
    st.write(df)

    # cast market_value to numeric, errors='coerce' will convert non-numeric values to NaN
    df["market_value"] = pd.to_numeric(df["market_value"], errors='coerce')

    df.to_excel("soccerdonna_market_values.xlsx", index=False)


def get_spielberichte_links(base_url="https://www.transfermarkt.de", schedule_url="https://www.transfermarkt.de/3-liga/gesamtspielplan/wettbewerb/L3?saison_id=2023&spieltagVon=1&spieltagBis=38"):
    res = requests.get(schedule_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    spielberichte_links = []
    for link in links:
        href = link['href']
        if "spielbericht" in href:
            full_url = base_url + href.split('?')[0]
            if full_url not in spielberichte_links:
                spielberichte_links.append(full_url)
    return spielberichte_links


def scrape_transfermarkt():
    spielberichte_links = get_spielberichte_links()
    st.write("spielberichte_links")
    st.write(spielberichte_links)
    dfs = []
    import defensive_network.utility.general
    for spielberichte_url in defensive_network.utility.general.progress_bar(spielberichte_links):
        st.write(f"Processing {spielberichte_url}")
        df = scrape_spielbericht(spielberichte_url)
        dfs.append(df)
        # Save or process the DataFrame as needed
        # For example, you can save it to a CSV file:
        # df.to_csv("market_values.csv", mode='a', header=False, index=False)
    # main()
    df = pd.concat(dfs, ignore_index=True)
    st.write("df")
    st.write(df)
    df.to_excel("transfermarkt_market_values.xlsx", index=False)


def main_transfermarkt():
    if st.toggle("Scrape Transfermarkt", False):
        scrape_transfermarkt()
    if st.toggle("Process Transfermarkt Data", True):
        process_transfermarkt()


def process_transfermarkt():
    df = pd.read_excel("transfermarkt_market_values.xlsx")
    df["name"] = df["name"].str.replace("-\n", "").str.strip()
    reference_date = pd.to_datetime("2024-01-19")
    df["difference_to_reference"] = (reference_date - df["date"]).abs().dt.days
    # drop duplicates but keep the closest date to the reference date
    df = df.sort_values(by=["name", "difference_to_reference"]).drop_duplicates(["name"]).reset_index(drop=True).drop(columns=["difference_to_reference"])
    st.write("df")
    st.write(df)
    df.to_excel("processed_transfermarkt_market_values.xlsx", index=False)


if __name__ == '__main__':
    main_transfermarkt()
