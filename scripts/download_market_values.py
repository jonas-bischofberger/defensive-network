import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

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
#         print(href)
#         if "spielbericht_" in href:
#             full_url = base_url + href.split('?')[0]
#             if full_url not in player_links:
#                 player_links.append(full_url)
#     return player_links


def scrape_spielbericht(url):
    url = url.replace("index", "aufstellung")
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    all_data = []
    pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows in the DataFrame4
    pd.set_option('display.width', 1000)  # Set a wider display width
    dfs = pd.read_html(res.text, header=0)
    print(f"Scraping {url}")
    for df in dfs:
        for _, row in df.iterrows():
            row_str = str(row)
            # print("row_str", row_str)
            if "MW" in row_str:
                player_name = row_str.split(".2")[-1].split(",")[0].strip()
                print("'" + player_name + "'")
                market_value = row_str.split("MW:")[-1].split(" €")[0].strip().replace(".", "")
                print(market_value, "€")
                player_data = {
                    'name': player_name,
                    'market_value': market_value,
                    'url': url
                }
                all_data.append(player_data)
    df = pd.DataFrame(all_data)
    df["market_value"] = df["market_value"].astype(float, errors='ignore')
    print("df")
    print(df)
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
            # print(f"Collected: {player_data['name']} ({player_data['overall']})")
            # time.sleep(1.5)
    df = pd.concat(data, ignore_index=True)
    print(df)

    # cast market_value to numeric, errors='coerce' will convert non-numeric values to NaN
    df["market_value"] = pd.to_numeric(df["market_value"], errors='coerce')

    df.to_excel("soccerdonna_market_values.xlsx", index=False)


if __name__ == '__main__':
    main()
