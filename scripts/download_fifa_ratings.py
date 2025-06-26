import requests
from bs4 import BeautifulSoup
import time

import defensive_network.parse.drive

base_url = "https://www.futbin.com"
league_url_template = "https://www.futbin.com/24/leagues/2215/gpfbl?page={page}&version=gold%2Csilver%2Cbronze"
headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_player_links(page):
    url = league_url_template.format(page=page)
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    player_links = []
    for link in links:
        href = link['href']
        if href.startswith("/24/player/"):
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
        club_img = soup.find("img", alt="Club")
        if club_img:
            return club_img.get("title", "N/A").strip()
        return None

    def _get_player_name(soup):
        """
        Extract the player name from the div with class 'playercard-24 playercard-l' using the title attribute.
        """
        player_div = soup.find("div", class_="playercard-24 playercard-l")
        if player_div:
            return player_div.get("title", "N/A").strip()
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
        return None

    def _get_position(soup):
        """
        Extract the player's position from the div with class 'playercard-24-position'.
        """
        pos_div = soup.find("div", class_="playercard-24-position")
        if pos_div:
            return pos_div.get_text(strip=True)
        return None

    rating = _get_overall_rating(soup)
    print(f"Overall Rating: {rating}")

    position = _get_position(soup)
    print(f"Position: {position}")

    def_awareness = _get_stat_by_id(soup, 35)  # Example stat ID for Def. Awareness
    interception = _get_stat_by_id(soup, 33)
    defending = _get_stat_by_id(soup, 5)
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


    # Main scraping loop
    all_players = []
    seen_links = set()
    page = 1

    while True:
        print(f"Scraping page {page}...")
        links = get_player_links(page)
        if not links:
            print("No more player links found. Stopping.")
            break

        print(f"Found {len(links)} player links on page {page}.")

        for link in links:
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
    df.to_csv("fifa_player_ratings.csv", index=False)
    defensive_network.parse.drive.upload_csv_to_drive(df, "fifa_ratings.csv")


if __name__ == '__main__':
    main()
