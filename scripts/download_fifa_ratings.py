import requests
import tqdm
from bs4 import BeautifulSoup
import time
import pandas as pd
import streamlit as st

import sys
import os
sys.path.append(os.path.join(__file__, '../..'))

import defensive_network.parse.drive

base_url = "https://www.futbin.com"
# base_url = "https://www.futbin.com"
league_url_template = "https://www.futbin.com/24/leagues/2076/3-liga?page={page}&version=gold%2Csilver%2Cbronze"
# league_url_template = "https://www.futbin.com/22/leagues/Major+League+Soccer+%28MLS%29?page={page}"
# league_url_template = "https://www.futbin.com/24/leagues/2215/gpfbl?page={page}&version=gold%2Csilver%2Cbronze"
headers = {
    "User-Agent": "Mozilla/5.0"
}
# https://www.futbin.com/22/leagues/3.+Liga+%28GER+3%29

def get_league_links(league_base_urls=("https://www.futbin.com/22/leagues", "https://www.futbin.com/22/leagues?page=2"), league_substring="/22/leagues/"):
    """
    >>> get_league_links(["https://www.futbin.com/22/nations"], league_substring="/22/nations/")
    ['https://www.futbin.com/22/nations/149/afghanistan', 'https://www.futbin.com/22/nations/1/albania', 'https://www.futbin.com/22/nations/97/algeria', 'https://www.futbin.com/22/nations/194/american%20samoa', 'https://www.futbin.com/22/nations/2/andorra', 'https://www.futbin.com/22/nations/98/angola', 'https://www.futbin.com/22/nations/62/anguilla', 'https://www.futbin.com/22/nations/63/antigua%20and%20barbuda', 'https://www.futbin.com/22/nations/52/argentina', 'https://www.futbin.com/22/nations/3/armenia', 'https://www.futbin.com/22/nations/64/aruba', 'https://www.futbin.com/22/nations/195/australia', 'https://www.futbin.com/22/nations/4/austria', 'https://www.futbin.com/22/nations/5/azerbaijan', 'https://www.futbin.com/22/nations/65/bahamas', 'https://www.futbin.com/22/nations/150/bahrain', 'https://www.futbin.com/22/nations/151/bangladesh', 'https://www.futbin.com/22/nations/66/barbados', 'https://www.futbin.com/22/nations/6/belarus', 'https://www.futbin.com/22/nations/7/belgium', 'https://www.futbin.com/22/nations/67/belize', 'https://www.futbin.com/22/nations/99/benin', 'https://www.futbin.com/22/nations/68/bermuda', 'https://www.futbin.com/22/nations/152/bhutan', 'https://www.futbin.com/22/nations/53/bolivia', 'https://www.futbin.com/22/nations/8/bosnia%20and%20herzegovina', 'https://www.futbin.com/22/nations/100/botswana', 'https://www.futbin.com/22/nations/54/brazil', 'https://www.futbin.com/22/nations/69/british%20virgin%20islands', 'https://www.futbin.com/22/nations/153/brunei%20darussalam', 'https://www.futbin.com/22/nations/9/bulgaria', 'https://www.futbin.com/22/nations/101/burkina%20faso', 'https://www.futbin.com/22/nations/102/burundi', 'https://www.futbin.com/22/nations/154/cambodia', 'https://www.futbin.com/22/nations/103/cameroon', 'https://www.futbin.com/22/nations/70/canada', 'https://www.futbin.com/22/nations/104/cape%20verde%20islands', 'https://www.futbin.com/22/nations/71/cayman%20islands', 'https://www.futbin.com/22/nations/105/central%20african%20republic', 'https://www.futbin.com/22/nations/106/chad', 'https://www.futbin.com/22/nations/55/chile', 'https://www.futbin.com/22/nations/155/china%20pr', 'https://www.futbin.com/22/nations/213/chinese%20taipei', 'https://www.futbin.com/22/nations/56/colombia', 'https://www.futbin.com/22/nations/214/comoros', 'https://www.futbin.com/22/nations/107/congo', 'https://www.futbin.com/22/nations/110/congo%20dr', 'https://www.futbin.com/22/nations/196/cook%20islands', 'https://www.futbin.com/22/nations/72/costa%20rica', 'https://www.futbin.com/22/nations/10/croatia', 'https://www.futbin.com/22/nations/73/cuba', 'https://www.futbin.com/22/nations/85/cura%C3%A7ao', 'https://www.futbin.com/22/nations/11/cyprus', 'https://www.futbin.com/22/nations/12/czechia', "https://www.futbin.com/22/nations/108/c%C3%B4te%20d'ivoire", 'https://www.futbin.com/22/nations/13/denmark', 'https://www.futbin.com/22/nations/109/djibouti', 'https://www.futbin.com/22/nations/74/dominica', 'https://www.futbin.com/22/nations/207/dominican%20republic', 'https://www.futbin.com/22/nations/57/ecuador', 'https://www.futbin.com/22/nations/111/egypt', 'https://www.futbin.com/22/nations/76/el%20salvador', 'https://www.futbin.com/22/nations/14/england', 'https://www.futbin.com/22/nations/112/equatorial%20guinea', 'https://www.futbin.com/22/nations/113/eritrea', 'https://www.futbin.com/22/nations/208/estonia', 'https://www.futbin.com/22/nations/142/eswatini', 'https://www.futbin.com/22/nations/114/ethiopia', 'https://www.futbin.com/22/nations/19/fyr%20macedonia', 'https://www.futbin.com/22/nations/16/faroe%20islands', 'https://www.futbin.com/22/nations/197/fiji', 'https://www.futbin.com/22/nations/17/finland', 'https://www.futbin.com/22/nations/18/france', 'https://www.futbin.com/22/nations/115/gabon', 'https://www.futbin.com/22/nations/116/gambia', 'https://www.futbin.com/22/nations/20/georgia', 'https://www.futbin.com/22/nations/21/germany', 'https://www.futbin.com/22/nations/117/ghana', 'https://www.futbin.com/22/nations/205/gibraltar', 'https://www.futbin.com/22/nations/22/greece', 'https://www.futbin.com/22/nations/206/greenland', 'https://www.futbin.com/22/nations/77/grenada', 'https://www.futbin.com/22/nations/157/guam', 'https://www.futbin.com/22/nations/78/guatemala', 'https://www.futbin.com/22/nations/118/guinea', 'https://www.futbin.com/22/nations/119/guinea-bissau', 'https://www.futbin.com/22/nations/79/guyana', 'https://www.futbin.com/22/nations/80/haiti', 'https://www.futbin.com/22/nations/81/honduras', 'https://www.futbin.com/22/nations/158/hong%20kong', 'https://www.futbin.com/22/nations/23/hungary', 'https://www.futbin.com/22/nations/24/iceland', 'https://www.futbin.com/22/nations/159/india', 'https://www.futbin.com/22/nations/160/indonesia', 'https://www.futbin.com/22/nations/75/international', 'https://www.futbin.com/22/nations/161/iran', 'https://www.futbin.com/22/nations/162/iraq', 'https://www.futbin.com/22/nations/26/israel', 'https://www.futbin.com/22/nations/27/italy', 'https://www.futbin.com/22/nations/82/jamaica', 'https://www.futbin.com/22/nations/163/japan', 'https://www.futbin.com/22/nations/164/jordan', 'https://www.futbin.com/22/nations/165/kazakhstan', 'https://www.futbin.com/22/nations/120/kenya', 'https://www.futbin.com/22/nations/166/korea%20dpr', 'https://www.futbin.com/22/nations/167/korea%20republic', 'https://www.futbin.com/22/nations/219/kosovo', 'https://www.futbin.com/22/nations/168/kuwait', 'https://www.futbin.com/22/nations/169/kyrgyzstan', 'https://www.futbin.com/22/nations/170/laos', 'https://www.futbin.com/22/nations/28/latvia', 'https://www.futbin.com/22/nations/171/lebanon', 'https://www.futbin.com/22/nations/121/lesotho', 'https://www.futbin.com/22/nations/122/liberia', 'https://www.futbin.com/22/nations/123/libya', 'https://www.futbin.com/22/nations/29/liechtenstein', 'https://www.futbin.com/22/nations/30/lithuania', 'https://www.futbin.com/22/nations/31/luxembourg', 'https://www.futbin.com/22/nations/172/macau', 'https://www.futbin.com/22/nations/124/madagascar', 'https://www.futbin.com/22/nations/125/malawi', 'https://www.futbin.com/22/nations/173/malaysia', 'https://www.futbin.com/22/nations/174/maldives', 'https://www.futbin.com/22/nations/126/mali', 'https://www.futbin.com/22/nations/32/malta', 'https://www.futbin.com/22/nations/127/mauritania', 'https://www.futbin.com/22/nations/128/mauritius', 'https://www.futbin.com/22/nations/83/mexico', 'https://www.futbin.com/22/nations/33/moldova', 'https://www.futbin.com/22/nations/175/mongolia', 'https://www.futbin.com/22/nations/15/montenegro', 'https://www.futbin.com/22/nations/84/montserrat', 'https://www.futbin.com/22/nations/129/morocco', 'https://www.futbin.com/22/nations/130/mozambique', 'https://www.futbin.com/22/nations/176/myanmar', 'https://www.futbin.com/22/nations/131/namibia', 'https://www.futbin.com/22/nations/177/nepal', 'https://www.futbin.com/22/nations/34/netherlands', 'https://www.futbin.com/22/nations/215/new%20caledonia', 'https://www.futbin.com/22/nations/198/new%20zealand', 'https://www.futbin.com/22/nations/86/nicaragua', 'https://www.futbin.com/22/nations/132/niger', 'https://www.futbin.com/22/nations/133/nigeria', 'https://www.futbin.com/22/nations/35/northern%20ireland', 'https://www.futbin.com/22/nations/36/norway', 'https://www.futbin.com/22/nations/178/oman', 'https://www.futbin.com/22/nations/179/pakistan', 'https://www.futbin.com/22/nations/180/palestine', 'https://www.futbin.com/22/nations/87/panama', 'https://www.futbin.com/22/nations/199/papua%20new%20guinea', 'https://www.futbin.com/22/nations/58/paraguay', 'https://www.futbin.com/22/nations/59/peru', 'https://www.futbin.com/22/nations/181/philippines', 'https://www.futbin.com/22/nations/37/poland', 'https://www.futbin.com/22/nations/38/portugal', 'https://www.futbin.com/22/nations/88/puerto%20rico', 'https://www.futbin.com/22/nations/182/qatar', 'https://www.futbin.com/22/nations/25/republic%20of%20ireland', 'https://www.futbin.com/22/nations/39/romania', 'https://www.futbin.com/22/nations/40/russia', 'https://www.futbin.com/22/nations/134/rwanda', 'https://www.futbin.com/22/nations/200/samoa', 'https://www.futbin.com/22/nations/41/san%20marino', 'https://www.futbin.com/22/nations/183/saudi%20arabia', 'https://www.futbin.com/22/nations/42/scotland', 'https://www.futbin.com/22/nations/136/senegal', 'https://www.futbin.com/22/nations/51/serbia', 'https://www.futbin.com/22/nations/137/seychelles', 'https://www.futbin.com/22/nations/138/sierra%20leone', 'https://www.futbin.com/22/nations/184/singapore', 'https://www.futbin.com/22/nations/43/slovakia', 'https://www.futbin.com/22/nations/44/slovenia', 'https://www.futbin.com/22/nations/201/solomon%20islands', 'https://www.futbin.com/22/nations/139/somalia', 'https://www.futbin.com/22/nations/140/south%20africa', 'https://www.futbin.com/22/nations/218/south%20sudan', 'https://www.futbin.com/22/nations/45/spain', 'https://www.futbin.com/22/nations/185/sri%20lanka', 'https://www.futbin.com/22/nations/89/st.%20kitts%20and%20nevis', 'https://www.futbin.com/22/nations/90/st.%20lucia', 'https://www.futbin.com/22/nations/91/st.%20vincent%20and%20the%20grenadines', 'https://www.futbin.com/22/nations/141/sudan', 'https://www.futbin.com/22/nations/92/suriname', 'https://www.futbin.com/22/nations/46/sweden', 'https://www.futbin.com/22/nations/47/switzerland', 'https://www.futbin.com/22/nations/186/syria', 'https://www.futbin.com/22/nations/135/s%C3%A3o%20tom%C3%A9%20e%20pr%C3%ADncipe', 'https://www.futbin.com/22/nations/202/tahiti', 'https://www.futbin.com/22/nations/187/tajikistan', 'https://www.futbin.com/22/nations/143/tanzania', 'https://www.futbin.com/22/nations/188/thailand', 'https://www.futbin.com/22/nations/212/timor-leste', 'https://www.futbin.com/22/nations/144/togo', 'https://www.futbin.com/22/nations/203/tonga', 'https://www.futbin.com/22/nations/93/trinidad%20and%20tobago', 'https://www.futbin.com/22/nations/145/tunisia', 'https://www.futbin.com/22/nations/189/turkmenistan', 'https://www.futbin.com/22/nations/94/turks%20and%20caicos%20islands', 'https://www.futbin.com/22/nations/48/t%C3%BCrkiye', 'https://www.futbin.com/22/nations/96/us%20virgin%20islands', 'https://www.futbin.com/22/nations/146/uganda', 'https://www.futbin.com/22/nations/49/ukraine', 'https://www.futbin.com/22/nations/190/united%20arab%20emirates', 'https://www.futbin.com/22/nations/95/united%20states', 'https://www.futbin.com/22/nations/60/uruguay', 'https://www.futbin.com/22/nations/191/uzbekistan', 'https://www.futbin.com/22/nations/204/vanuatu', 'https://www.futbin.com/22/nations/61/venezuela', 'https://www.futbin.com/22/nations/192/vietnam', 'https://www.futbin.com/22/nations/50/wales', 'https://www.futbin.com/22/nations/193/yemen', 'https://www.futbin.com/22/nations/147/zambia', 'https://www.futbin.com/22/nations/148/zimbabwe']
    >>> get_league_links()
    ['https://www.futbin.com/21/leagues/1A+Pro+League+%28BEL+1%29', 'https://www.futbin.com/21/leagues/3.+Liga+%28GER+3%29', 'https://www.futbin.com/21/leagues/3F+Superliga+%28DEN+1%29', 'https://www.futbin.com/21/leagues/A-League+%28AUS+1%29', 'https://www.futbin.com/21/leagues/Allsvenskan+%28SWE+1%29', 'https://www.futbin.com/21/leagues/Bundesliga', 'https://www.futbin.com/21/leagues/Bundesliga+2+%28GER+2%29', 'https://www.futbin.com/21/leagues/Calcio+B+%28ITA+2%29', 'https://www.futbin.com/21/leagues/%C4%8Cesk%C3%A1+Liga+%28CZE+1%29', 'https://www.futbin.com/21/leagues/Chinese+FA+Super+L.+%28CHN+1%29', 'https://www.futbin.com/21/leagues/CONMEBOL+Libertadores', 'https://www.futbin.com/21/leagues/CONMEBOL+Sudamericana', 'https://www.futbin.com/21/leagues/EFL+Championship+%28ENG+2%29', 'https://www.futbin.com/21/leagues/EFL+League+One+%28ENG+3%29', 'https://www.futbin.com/21/leagues/EFL+League+Two+%28ENG+4%29', 'https://www.futbin.com/21/leagues/Eliteserien+%28NOR+1%29', 'https://www.futbin.com/21/leagues/Eredivisie+%28NED+1%29', 'https://www.futbin.com/21/leagues/Finnliiga+%28FIN+1%29', 'https://www.futbin.com/21/leagues/Hellas+Liga+%28GRE+1%29', 'https://www.futbin.com/21/leagues/Icons', 'https://www.futbin.com/21/leagues/K+League+1+%28KOR+1%29', 'https://www.futbin.com/21/leagues/LaLiga+Santander', 'https://www.futbin.com/21/leagues/LaLiga+SmartBank+%28ESP+2%29', 'https://www.futbin.com/21/leagues/League+of+Russia+%28RUS+1%29', 'https://www.futbin.com/21/leagues/LIGA+BBVA+MX+%28MEX+1%29', 'https://www.futbin.com/21/leagues/Liga+Hrvatska+%28CRO+1%29', 'https://www.futbin.com/21/leagues/Liga+I+%28ROM+1%29', 'https://www.futbin.com/21/leagues/Liga+NOS+%28POR+1%29', 'https://www.futbin.com/21/leagues/Ligue+1+Conforama', 'https://www.futbin.com/21/leagues/Ligue+2+%28FRA+2%29']
    """
    league_links = []
    for league_base_url in league_base_urls:
        # get all links that begin with https://www.futbin.com/24/leagues/
        res = requests.get(league_base_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.find_all("a", href=True)
        for link in links:
            href = link['href']
            if league_substring in href:
                full_url = base_url + href.split('?')[0]
                if full_url not in league_links:
                    league_links.append(full_url)

    return league_links


def get_player_links(page, league_url_template=league_url_template):
    url = league_url_template.format(page=page)
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    player_links = []
    for link in links:
        href = link['href']
        if "/player/" in href:
            full_url = base_url + href.split('?')[0]
            if full_url not in player_links:
                player_links.append(full_url)
    return player_links


def scrape_player_data(url):
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    def _get_stat_by_id(soup, stat_id):
        """
        Given a BeautifulSoup object and a stat ID, return the stat value.
        """
        stat_div = soup.find("div", class_="player-stat-value-wrapper", attrs={"data-stat-id": str(stat_id)})
        if stat_div:
            value_div = stat_div.find("div", class_="player-stat-value")
            if value_div:
                return value_div.get("data-stat-value", value_div.text).strip()
        return None

    def _get_club_name(soup):
        """
        Extract the club name from the player's profile page by looking for the img tag with alt='Club'.
        """
        # <img alt="club" class="info_club" src="https://cdn.futbin.com/content/fifa21/img/clubs/232.png">
        club_img = soup.find("img", alt="Club")
        if club_img:
            return club_img.get("title", "N/A").strip()

        # Loop over all <tr> tags and find the one with <th> text "Club"
        for tr in soup.find_all('tr'):
            th = tr.find('th')
            if th and th.text.strip() == "Club":
                td = tr.find('td', class_='table-row-text')
                if td:
                    club_name = td.find('a').text.strip()
                    return club_name
                break
        return None

    def _get_player_name(soup):
        """
        Extract the player name from the div with class 'playercard-24 playercard-l' using the title attribute.
        """
        player_div = soup.find("div", class_="playercard-24 playercard-l")
        if player_div:
            return player_div.get("title", "N/A").strip()

        for tr in soup.find_all('tr'):
            th = tr.find('th')
            if th and th.text.strip() == "Name":
                td = tr.find('td', class_='table-row-text')
                if td:
                    player_name = td.text.strip()
                    return player_name
                break

        # Find the <tr> where the <th> text is "Name"
        for tr in soup.find_all('tr'):
            th = tr.find('th')
            if th and th.text.strip() == "Name":
                td = tr.find('td')
                if td:
                    player_name = td.text.strip()
                    print(player_name)
                    return player_name
                break
        return None

    def _get_country_name(soup):
        """
        Extract the player's nationality from the img tag with alt='Nation'.
        """
        nation_img = soup.find("img", alt="Nation")
        if nation_img:
            return nation_img.get("title", "N/A").strip()
        return None

    def _get_overall_rating(soup):
        """
        Extract the player's overall rating from the div with class 'playercard-24-rating'.
        """
        rating_div = soup.find("div", class_="playercard-24-rating")
        if rating_div:
            return rating_div.get_text(strip=True)
        rating_div = soup.find("div", class_="playercard-23-rating")
        if rating_div:
            return rating_div.get_text(strip=True)
        return None

    def _get_position(soup):
        """
        Extract the player's position from the div with class 'playercard-24-position'.
        """
        pos_div = soup.find("div", class_="playercard-24-position")
        if pos_div:
            return pos_div.get_text(strip=True)
        pos_div = soup.find("div", class_="playercard-23-position")
        if pos_div:
            return pos_div.get_text(strip=True)
        return None

    def get_def_val_21(soup, identifier="main-defending-val-0"):
        main_div = soup.find('div', id=identifier)
        inner_val = main_div.find('div', class_='stat_val')
        number = int(inner_val.get_text(strip=True))
        return number

    rating = _get_overall_rating(soup)
    print(f"Overall Rating: {rating}")

    position = _get_position(soup)
    print(f"Position: {position}")

    def_awareness = _get_stat_by_id(soup, 35)  # Example stat ID for Def. Awareness
    # def_awareness = get_def_val_21(soup, "sub-marking-val-0")  # Example stat ID for Def. Awareness
    interception = _get_stat_by_id(soup, 33)
    # interception = get_def_val_21(soup, "sub-interceptions-val-0")
    defending = _get_stat_by_id(soup, 5)
    # defending = get_def_val_21(soup, "main-defending-val-0")
    print(f"Def. Awareness: {def_awareness}")
    print(f"Interceptions: {interception}")
    print(f"Defending: {defending}")
    club_name = _get_club_name(soup)
    player_name = _get_player_name(soup)
    country = _get_country_name(soup)
    print(f"Club: {club_name}")
    print(f"Player Name: {player_name}")
    print(f"Country: {country}")

    return {
        "name": player_name,
        "position": position,
        "overall": rating,
        "club": club_name,
        "country": country,
        "defending": defending,
        "def_awareness": def_awareness,
        "interceptions": interception,
        "url": url
    }


def main():
    df_existing = defensive_network.parse.drive.download_csv_from_drive("fifa_ratings.csv")
    urls = df_existing["url"].tolist()

    # Main scraping loop
    all_players = []
    seen_links = set()
    use_league_links = True
    league_links = get_league_links(["https://www.futbin.com/22/nations"], league_substring="/22/nations/")

    if use_league_links:
        league_url_templates = [f"{link}?page={{page}}&version=gold,silver,bronze" for link in league_links]
    else:
        league_url_templates = [league_url_template]

    print("league_url_templates")
    print(league_url_templates)

    for llt in tqdm.tqdm(league_url_templates, total=len(league_url_templates), desc="Collecting FIFA ratings"):
        page = 1

        while True:
            print(f"Scraping page {page}...")
            links = get_player_links(page, league_url_template=llt)
            if not links:
                print("No more player links found. Stopping.")
                break

            print(f"Found {len(links)} player links on page {page}: {links[:5]}... (total {len(links)})")

            for link in links:
                if link in urls:
                    print(f"Skipping already seen link: {link}")
                    continue
                if link not in seen_links:
                    seen_links.add(link)
                    player_data = scrape_player_data(link)
                    all_players.append(player_data)
                    print(f"Collected: {player_data['name']} ({player_data['overall']})")
                    # time.sleep(1.5)

            page += 1
            # time.sleep(2)

        # Print results
        print(f"\nScraped {len(all_players)} players.\n")
        for p in all_players:
            print(p)

        import pandas as pd
        df = pd.DataFrame(all_players)
        df["comp"] = "FIFA Men's World Cup"
        df.to_csv("fifa_player_ratings.csv", index=False)
        defensive_network.parse.drive.append_to_parquet_on_drive(df, "fifa_ratings.csv", key_cols=["url"], format="csv")


def add_extra_data():
    # hand-collected
    data = [
        {"name": "Toby Alderweireld", "position": "CB", "overall": 83, "club": "Belgium", "country": "Belgium", "defending": 86, "def_awareness": 87, "interceptions": 85, "url": "https://www.futwiz.com/en/fifa22/career-mode/player/toby-alderweireld/259"},
        # Lovren RUS, no exact data
        {"name": "Boualem Khoukhi", "position": "CB", "overall": 68, "club": "Qatar", "country": "Qatar", "defending": 70, "def_awareness": 71, "interceptions": 71, "url": "https://sofifa.com/player/11627981/boualem-khoukhi/"},
        {"name": "Yeltsin Tejeda", "position": "CM", "overall": 68, "club": "Costa Rica", "country": "Costa Rica", "defending": 66, "def_awareness": 67, "interceptions": 67, "url": "https://www.fifaindex.com/es/player/216810/yeltsin-tejeda/fifa22_555/"},  # Defending unclear
        {"name": "Homam Ahmed", "position": "LB", "overall": 68, "club": "Qatar", "country": "Qatar", "defending": 64, "def_awareness": 63, "interceptions": 62, "url": "https://sofifa.com/player/268776/homam-ahmed/230003?hl=en-US"},  # Defending unclear, FIFA 23
        {"name": "Woo-young Jung", "position": "CM", "overall": 69, "club": "South Korea", "country": "South Korea", "defending": 62, "def_awareness": 64, "interceptions": 68, "url": "https://sofifa.com/player/211003/woo-young-jung/240002/?hl=en-US"},  # Defending unclear, FIFA 23
        {"name": "Yahia Attiyat-Allah", "position": "LB", "overall": 73, "club": "Morocco", "country": "Morocco", "defending": 68, "def_awareness": 67, "interceptions": 68, "url": "https://sofifa.com/player/268866/yahya-attiat-allah/250007?hl=en-US"},  # Defending unclear, FIFA 23
        {"name": "Mitchell Duke", "position": "CF", "overall": 67, "club": "Australia", "country": "Australia", "defending": 50, "def_awareness": 47, "interceptions": 51, "url": "https://sofifa.com/player/209885/mitchell-duke/220069?hl=en-US"},  # Defending unclear, FIFA 23
        {"name": "Jakub Kiwior", "position": "CB", "overall": 71, "club": "Poland", "country": "Poland", "defending": 70, "def_awareness": 70, "interceptions": 71, "url": "https://www.futbin.com/23/player/47067/jakub-kiwior"},  # Defending unclear, FIFA 23
        {"name": "In-beom Hwang", "position": "CM", "overall": 72, "club": "South Korea", "country": "South Korea", "defending": 52, "def_awareness": 53, "interceptions": 59, "url": "https://sofifa.com/player/228010/in-beom-hwang/220061?hl=en-US"},  # Defending unclear, FIFA 23
        {"name": "Gue-sung Cho", "position": "CF", "overall": 72, "club": "South Korea", "country": "South Korea", "defending": 24, "def_awareness": 30, "interceptions": 23, "url": "https://sofifa.com/player/247686/gue-sung-cho/240002?hl=en-US"},  # Defending unclear, FIFA 23
        {"name": "Abdelkarim Hassan", "position": "CB", "overall": 68, "club": "Qatar", "country": "Qatar", "defending": 70, "def_awareness": 69, "interceptions": 68, "url": "https://sofifa.com/player/11628102/abdelkarim-hassan/"},  # Defending unclear, FIFA 23
        {"name": "Montassar Talbi", "position": "CB", "overall": 65, "club": "Tunisia", "country": "Tunisia", "defending": 66, "def_awareness": 64, "interceptions": 64, "url": "https://www.futwiz.com/en/fifa22/career-mode/player/montassar-talbi/14414"},  # Defending unclear, FIFA 23
    ]
    df = pd.DataFrame(data)
    df["comp"] = "FIFA Men's World Cup"
    st.write("df")
    st.write(df)

    df_existing = defensive_network.parse.drive.download_csv_from_drive("fifa_ratings.csv")
    st.write("df_existing")
    st.write(df_existing)

    defensive_network.parse.drive.append_to_parquet_on_drive(df, "fifa_ratings.csv", key_cols=["url"], format="csv")
    df_existing = defensive_network.parse.drive.download_csv_from_drive("fifa_ratings.csv")
    st.write("df_existing")
    st.write(df_existing)


if __name__ == '__main__':
    add_extra_data()
