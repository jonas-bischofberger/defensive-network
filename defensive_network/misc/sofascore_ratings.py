import pandas as pd
import selenium.webdriver
import selenium.webdriver.chrome.service
import tqdm
import webdriver_manager.chrome
from selenium.webdriver.common.by import By


def uniquify_keep_order(lst):
    return list(dict.fromkeys(lst))


def get_chrome_driver():
    options = selenium.webdriver.ChromeOptions()
    options.add_argument("--disable-search-engine-choice-screen")  # otherwise you get a stupid "What's your default search engine?" popup at the beginning

    driver = selenium.webdriver.Chrome(
        service=selenium.webdriver.chrome.service.Service(webdriver_manager.chrome.ChromeDriverManager().install()),
        options=options
    )
    driver.implicitly_wait(10)
    return driver


def get_match_ratings(match_link, driver=None):
    print(f"Downloading from {match_link}...")
    data = {
        "player": [],
        "rating": [],
        "match_string": [],
        "match_link": [],
    }

    if driver is None:
        driver = get_chrome_driver()

    driver.get(match_link)

    # Get Match info
    match_details = driver.find_element(By.XPATH, "//ul[contains(@class, 'gWYkXa')]").text.split("\n")
    matchday = match_details[2].strip()
    match_string = match_details[3].replace("live score, H2H results, standings and prediction", "").strip()
    team1 = match_string.split(" vs ")[0]
    team2 = match_string.split(" vs ")[1]
    match_string = f"{matchday}: {team1} vs {team2}"

    # Get all elements that contain a Sofascore rating
    all_arias = driver.find_elements(By.XPATH, "//span[@aria-valuenow]")

    for arias in all_arias:
        # Extract rating
        try:
            rating = float(arias.get_attribute('aria-valuenow'))
        except ValueError:
            rating = None

        parent = arias.find_element(By.XPATH, "../../../..")
        # print("parent")
        # print(parent.get_attribute('innerHTML'))

        # Skip team ratings
        if "Team" in parent.get_attribute('innerHTML') or "_formation" in parent.get_attribute('innerHTML'):
            continue

        # Get player name for different types of elements (subs, starters, etc.)
        try:
            if "title" not in parent.get_attribute('innerHTML'):
                raise Exception("No title found")
            elif "Box emTNBQ" in parent.get_attribute("innerHTML"):
                player_name = parent.find_element(By.XPATH, ".//div[@title]").get_attribute('title')
            else:
                player_name = parent.find_element(By.XPATH, ".//span[@title]").get_attribute('title')
        except Exception as e:
            player_name = parent.find_element(By.XPATH, ".//span[@class]").get_attribute('innerHTML')

        player_name = player_name.replace("(c)", "").strip()  # captain
        player_name = player_name.replace('­', '')  # names sometimes have a ZWNJ character

        data["player"].append(player_name)
        data["rating"].append(rating)
        data["match_string"].append(match_string)
        data["match_link"].append(match_link)

    df = pd.DataFrame(data)

    df = df.drop_duplicates().reset_index(drop=True)
    print("df")
    print(df)

    # assert every player has only 1 rating
    assert len(df["player"].unique()) == len(df)

    return df


def get_tournament_ratings(tournament_link, driver=None):
    # Get chromedriver for selenium
    if driver is None:
        driver = get_chrome_driver()

    ### 1. Call sofascore website of EURO 2024
    driver.get(tournament_link)

    ### 2. Accept cookies if necessary
    print("Waiting for cookie banner...")
    try:
        # constent_button = p_consent(driver)
        constent_button = driver.find_element(By.XPATH, "//*[text() = 'Einwilligen']")
        constent_button.click()
    except Exception as e:
        print(f"No cookie banner found: {type(e)}")

    ### 3. Collect links to all matches of the tournament
    print("Collecting links to matches...")
    links = driver.find_elements(By.TAG_NAME, 'a')
    substring = 'sofascore.com/football/match/'
    match_links = [link.get_attribute('href') for link in links if link.get_attribute('href') and substring in link.get_attribute('href')]

    # Click on the "Previous" button repeatedly until we have collected all match links
    previous_matching_links = []
    i = 0
    n_no_changes_max = 10
    n_no_changes = 0
    while True:
        i += 1
        prev_button = driver.find_element(By.XPATH, "//button[@class='Button iCnTrv']")
        driver.execute_script("arguments[0].click();", prev_button)

        new_links = driver.find_elements(By.TAG_NAME, 'a')
        new_matching_links = [link.get_attribute('href') for link in new_links if link.get_attribute('href') and substring in link.get_attribute('href')]

        match_links.extend(new_matching_links)
        match_links = uniquify_keep_order(match_links)

        print(f"Found {len(match_links)} matches after {i} clicks.")

        if len(previous_matching_links) == len(match_links):
            n_no_changes += 1
            if n_no_changes == n_no_changes_max:
                print(f"No more matches found. Found {len(match_links)} after {i} clicks.")
                break

        previous_matching_links = match_links.copy()

    match_links = uniquify_keep_order(match_links)

    ### 4. Download match data
    dfs = []
    for link in tqdm.tqdm(match_links):
        df = get_match_ratings(link, driver)
        dfs.append(df)

    # Build a single dataframe from all match data
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def main():
    # Make pandas output more readable
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    tournament_link = "https://www.sofascore.com/tournament/football/world/world-championship/16"
    # tournament_link = "https://www.sofascore.com/tournament/football/europe/european-championship/1"

    df = get_tournament_ratings(tournament_link)
    print("df")
    print(df)

    tournament_string = tournament_link.split("/")[-2]
    df.to_csv(f"sofascore_ratings_{tournament_string}.csv", index=False)
    df.to_excel(f"sofascore_ratings_{tournament_string}.xlsx", index=False)


if __name__ == '__main__':
    main()
