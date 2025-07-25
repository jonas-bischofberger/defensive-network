import requests
import tqdm
from bs4 import BeautifulSoup
import time

import defensive_network.parse.drive

base_url = "https://www.futbin.com"
# base_url = "https://www.futbin.com"
league_url_template = "https://www.futbin.com/24/leagues/2076/3-liga?page={page}&version=gold%2Csilver%2Cbronze"
league_url_template = "https://www.futbin.com/22/leagues/Major+League+Soccer+%28MLS%29?page={page}"
# league_url_template = "https://www.futbin.com/24/leagues/2215/gpfbl?page={page}&version=gold%2Csilver%2Cbronze"
headers = {
    "User-Agent": "Mozilla/5.0"
}
# https://www.futbin.com/22/leagues/3.+Liga+%28GER+3%29

def get_league_links(league_base_urls=["https://www.futbin.com/22/leagues", "https://www.futbin.com/22/leagues?page=2"]):
    """
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
            if "/22/leagues/" in href:
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
    league_links = get_league_links()

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


if __name__ == '__main__':
    main()
