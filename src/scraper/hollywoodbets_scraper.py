import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
import cloudscraper
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

from config import SCRAPING_CONFIG
from src.utils.logger import logger
from src.utils.database import db

class HollywoodBetsScraper:
    def __init__(self, use_selenium: bool = True):
        self.base_url = SCRAPING_CONFIG.HOLLYWOODBETS_URL
        self.headers = SCRAPING_CONFIG.HEADERS
        self.use_selenium = use_selenium
        self.scraper = cloudscraper.create_scraper()
        
        if use_selenium:
            self.driver = self._init_selenium()
    
    def _init_selenium(self) -> webdriver.Chrome:
        """Initialize Selenium WebDriver with options."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')
        
        # Add additional options to avoid detection
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Execute CDP commands to avoid detection
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": self.headers['User-Agent']
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[str]:
        """Make HTTP request with retries and delays."""
        for attempt in range(retries):
            try:
                time.sleep(random.uniform(1, 3))  # Random delay
                
                if self.use_selenium and 'race-cards' in url:
                    return self._get_with_selenium(url)
                else:
                    response = self.scraper.get(url, headers=self.headers)
                    response.raise_for_status()
                    
                    if 'Access Denied' in response.text:
                        logger.warning(f"Access denied for {url}, trying selenium...")
                        if self.use_selenium:
                            return self._get_with_selenium(url)
                    
                    return response.text
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(5 * (attempt + 1))  # Exponential backoff
        
        return None
    
    def _get_with_selenium(self, url: str) -> Optional[str]:
        """Get page content using Selenium."""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # Wait for JavaScript
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Selenium failed for {url}: {e}")
            return None
    
    def scrape_race_cards(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Scrape race cards for a specific date."""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        url = f"{SCRAPING_CONFIG.RACE_CARDS_URL}/{date_str}"
        
        logger.info(f"Scraping race cards for {date_str}")
        
        html = self._make_request(url)
        if not html:
            logger.error(f"Failed to get race cards for {date_str}")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        races = []
        
        try:
            # Find race meeting containers
            meetings = soup.find_all('div', class_=re.compile(r'race-meeting|meeting-card'))
            
            for meeting in meetings:
                meeting_name = self._extract_meeting_name(meeting)
                
                # Find races in this meeting
                race_elements = meeting.find_all('div', class_=re.compile(r'race-card|race-item'))
                
                for race_element in race_elements:
                    race_data = self._parse_race_card(race_element, date, meeting_name)
                    if race_data:
                        races.append(race_data)
                        logger.info(f"Found race: {race_data['race_name']} at {race_data['race_time']}")
        
        except Exception as e:
            logger.error(f"Error parsing race cards: {e}")
        
        return races
    
    def _extract_meeting_name(self, meeting_element) -> str:
        """Extract meeting/venue name."""
        try:
            # Look for various possible selectors
            name_element = meeting_element.find(['h2', 'h3', 'div'], class_=re.compile(r'meeting-name|venue'))
            if name_element:
                return name_element.text.strip()
            
            # Try data attributes
            if meeting_element.has_attr('data-venue'):
                return meeting_element['data-venue']
            
            return "Unknown Venue"
        except:
            return "Unknown Venue"
    
    def _parse_race_card(self, race_element, race_date: datetime, meeting_name: str) -> Optional[Dict[str, Any]]:
        """Parse individual race card."""
        try:
            # Extract race time
            time_element = race_element.find('div', class_=re.compile(r'race-time|time'))
            if not time_element:
                return None
            
            race_time_str = time_element.text.strip()
            race_time = self._parse_race_time(race_date, race_time_str)
            
            # Extract race number
            race_number = 1
            number_element = race_element.find('div', class_=re.compile(r'race-number|number'))
            if number_element:
                number_text = number_element.text.strip()
                numbers = re.findall(r'\d+', number_text)
                if numbers:
                    race_number = int(numbers[0])
            
            # Extract race name
            name_element = race_element.find('div', class_=re.compile(r'race-name|name'))
            race_name = name_element.text.strip() if name_element else f"Race {race_number}"
            
            # Extract race details
            details_element = race_element.find('div', class_=re.compile(r'race-details|details'))
            distance, race_class, prize_money = self._parse_race_details(details_element)
            
            # Generate unique race ID
            race_id = f"{race_date.strftime('%Y%m%d')}_{meeting_name.replace(' ', '_')}_R{race_number}"
            
            race_data = {
                'race_id': race_id,
                'meeting_id': f"{race_date.strftime('%Y%m%d')}_{meeting_name.replace(' ', '_')}",
                'race_date': race_date.date(),
                'race_time': race_time,
                'race_number': race_number,
                'race_name': race_name,
                'race_class': race_class,
                'distance': distance,
                'prize_money': prize_money,
                'going': 'Unknown',  # Will be updated from results
                'meeting_name': meeting_name
            }
            
            # Parse runners
            runners_table = race_element.find('table', class_=re.compile(r'runners|horses'))
            if runners_table:
                runners = self._parse_runners(runners_table, race_data)
                race_data['runners'] = runners
            
            return race_data
            
        except Exception as e:
            logger.error(f"Error parsing race card: {e}")
            return None
    
    def _parse_race_time(self, race_date: datetime, time_str: str) -> datetime:
        """Parse race time string into datetime."""
        try:
            # Remove any non-time characters
            time_str = re.sub(r'[^\d:APM\s]', '', time_str, flags=re.IGNORECASE)
            
            # Try different time formats
            time_formats = ['%I:%M %p', '%H:%M', '%I%p', '%H%M']
            
            for fmt in time_formats:
                try:
                    time_obj = datetime.strptime(time_str.strip(), fmt).time()
                    return datetime.combine(race_date.date(), time_obj)
                except:
                    continue
            
            # Default to 12:00 PM if parsing fails
            return datetime.combine(race_date.date(), datetime.strptime('12:00 PM', '%I:%M %p').time())
            
        except Exception as e:
            logger.error(f"Error parsing time {time_str}: {e}")
            return datetime.combine(race_date.date(), datetime.strptime('12:00 PM', '%I:%M %p').time())
    
    def _parse_race_details(self, details_element) -> tuple:
        """Parse race distance, class, and prize money."""
        distance = 0
        race_class = "Unknown"
        prize_money = 0.0
        
        if not details_element:
            return distance, race_class, prize_money
        
        details_text = details_element.text
        
        # Extract distance (in meters)
        distance_match = re.search(r'(\d+)\s*m', details_text, re.IGNORECASE)
        if distance_match:
            distance = int(distance_match.group(1))
        
        # Extract race class
        class_match = re.search(r'(Maiden|Class\s*\d+|Grade\s*[A-Z]+|Listed)', details_text, re.IGNORECASE)
        if class_match:
            race_class = class_match.group(1)
        
        # Extract prize money
        money_match = re.search(r'R\s*(\d+(?:,\d+)*(?:\.\d+)?)', details_text)
        if money_match:
            money_str = money_match.group(1).replace(',', '')
            prize_money = float(money_str)
        
        return distance, race_class, prize_money
    
    def _parse_runners(self, runners_table, race_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse runners/horses from the race."""
        runners = []
        
        try:
            rows = runners_table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 5:
                    continue
                
                # Extract horse information
                horse_cell = cells[1]  # Usually second cell contains horse name
                horse_link = horse_cell.find('a')
                horse_name = horse_link.text.strip() if horse_link else horse_cell.text.strip()
                
                # Generate horse ID
                horse_id = f"{race_data['race_id']}_{horse_name.replace(' ', '_')}"
                
                # Extract jockey and trainer
                jockey_name = cells[2].text.strip() if len(cells) > 2 else "Unknown"
                trainer_name = cells[3].text.strip() if len(cells) > 3 else "Unknown"
                
                # Extract weight
                weight_text = cells[4].text.strip() if len(cells) > 4 else ""
                weight = self._parse_weight(weight_text)
                
                # Extract draw
                draw = int(cells[0].text.strip()) if cells[0].text.strip().isdigit() else 0
                
                # Extract odds (if available)
                odds_cell = cells[-1] if cells else None
                odds = self._parse_odds(odds_cell.text if odds_cell else "")
                
                runner_data = {
                    'horse_id': horse_id,
                    'horse_name': horse_name,
                    'saddle_number': draw,
                    'jockey_name': jockey_name,
                    'trainer_name': trainer_name,
                    'weight': weight,
                    'draw': draw,
                    'odds': odds,
                    'form_rating': 0,  # Will be populated from form guide
                    'official_rating': 0,
                    'days_since_last_run': 0,
                    'career_starts': 0,
                    'career_wins': 0
                }
                
                runners.append(runner_data)
                
        except Exception as e:
            logger.error(f"Error parsing runners: {e}")
        
        return runners
    
    def _parse_weight(self, weight_text: str) -> float:
        """Parse weight string into kilograms."""
        try:
            # Remove non-numeric characters except decimal point
            weight_str = re.sub(r'[^\d.]', '', weight_text)
            if weight_str:
                return float(weight_str)
        except:
            pass
        return 0.0
    
    def _parse_odds(self, odds_text: str) -> float:
        """Parse odds string into decimal odds."""
        try:
            # Handle fractional odds (e.g., 5/2, 3/1)
            if '/' in odds_text:
                parts = odds_text.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    return (numerator / denominator) + 1
            
            # Handle decimal odds
            odds_match = re.search(r'(\d+\.?\d*)', odds_text)
            if odds_match:
                return float(odds_match.group(1))
            
        except:
            pass
        return 0.0
    
    def scrape_race_results(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Scrape race results for a specific date."""
        if date is None:
            date = datetime.now() - timedelta(days=1)  # Yesterday's results
        
        date_str = date.strftime('%Y-%m-%d')
        url = f"{SCRAPING_CONFIG.RESULTS_URL}/{date_str}"
        
        logger.info(f"Scraping race results for {date_str}")
        
        html = self._make_request(url)
        if not html:
            logger.error(f"Failed to get results for {date_str}")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        try:
            # Find result containers
            result_containers = soup.find_all('div', class_=re.compile(r'race-result|result-card'))
            
            for container in result_containers:
                result_data = self._parse_race_result(container, date)
                if result_data:
                    results.append(result_data)
                    logger.info(f"Found result for race: {result_data.get('race_name', 'Unknown')}")
        
        except Exception as e:
            logger.error(f"Error parsing race results: {e}")
        
        return results
    
    def _parse_race_result(self, result_element, race_date: datetime) -> Optional[Dict[str, Any]]:
        """Parse individual race result."""
        try:
            # Extract race information
            race_header = result_element.find('div', class_=re.compile(r'race-header|header'))
            if not race_header:
                return None
            
            # Extract race name and number
            race_text = race_header.text.strip()
            race_number_match = re.search(r'Race\s*(\d+)', race_text, re.IGNORECASE)
            race_number = int(race_number_match.group(1)) if race_number_match else 1
            
            race_name_match = re.search(r'(?:Race\s*\d+[:-\s]*)?(.+)', race_text)
            race_name = race_name_match.group(1).strip() if race_name_match else f"Race {race_number}"
            
            # Extract going and distance
            details_div = result_element.find('div', class_=re.compile(r'race-details|details'))
            going = "Unknown"
            distance = 0
            
            if details_div:
                details_text = details_div.text
                going_match = re.search(r'Going:\s*(\w+)', details_text, re.IGNORECASE)
                if going_match:
                    going = going_match.group(1)
                
                distance_match = re.search(r'(\d+)\s*m', details_text, re.IGNORECASE)
                if distance_match:
                    distance = int(distance_match.group(1))
            
            # Extract winning time
            time_div = result_element.find('div', class_=re.compile(r'winning-time|time'))
            winning_time = 0.0
            if time_div:
                time_text = time_div.text
                time_match = re.search(r'(\d+\.?\d*)\s*sec', time_text, re.IGNORECASE)
                if time_match:
                    winning_time = float(time_match.group(1))
            
            # Find results table
            results_table = result_element.find('table', class_=re.compile(r'results|finish'))
            if not results_table:
                return None
            
            # Parse finishing positions
            rows = results_table.find_all('tr')[1:]  # Skip header
            positions = {}
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue
                
                position = cells[0].text.strip()
                horse_cell = cells[1]
                horse_link = horse_cell.find('a')
                horse_name = horse_link.text.strip() if horse_link else horse_cell.text.strip()
                
                if position.isdigit():
                    pos_num = int(position)
                    positions[pos_num] = horse_name
            
            # Extract dividends
            dividends_div = result_element.find('div', class_=re.compile(r'dividends|payouts'))
            dividends = self._parse_dividends(dividends_div) if dividends_div else {}
            
            # Generate race ID to match with race card
            race_id = f"{race_date.strftime('%Y%m%d')}_R{race_number}"
            
            result_data = {
                'race_id': race_id,
                'race_name': race_name,
                'race_date': race_date.date(),
                'race_number': race_number,
                'distance': distance,
                'going': going,
                'winning_time': winning_time,
                'positions': positions,
                'dividends': dividends
            }
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error parsing race result: {e}")
            return None
    
    def _parse_dividends(self, dividends_element) -> Dict[str, Any]:
        """Parse dividend/payout information."""
        dividends = {}
        
        try:
            # Look for different types of dividends
            div_text = dividends_element.text
            
            # Win dividend
            win_match = re.search(r'Win:\s*R\s*(\d+\.?\d*)', div_text)
            if win_match:
                dividends['win'] = float(win_match.group(1))
            
            # Place dividend
            place_match = re.search(r'Place:\s*R\s*(\d+\.?\d*)', div_text)
            if place_match:
                dividends['place'] = float(place_match.group(1))
            
            # Quartet/Exacta etc.
            exacta_match = re.search(r'(?:Exacta|Quartet):\s*R\s*(\d+\.?\d*)', div_text)
            if exacta_match:
                dividends['exacta'] = float(exacta_match.group(1))
            
        except Exception as e:
            logger.error(f"Error parsing dividends: {e}")
        
        return dividends
    
    def scrape_tipsters(self) -> List[Dict[str, Any]]:
        """Scrape tipsters and their tips."""
        logger.info("Scraping tipsters...")
        
        html = self._make_request(SCRAPING_CONFIG.TIPSTERS_URL)
        if not html:
            logger.error("Failed to get tipsters page")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        tipsters = []
        
        try:
            # Find tipster containers
            tipster_containers = soup.find_all('div', class_=re.compile(r'tipster|expert'))
            
            for container in tipster_containers:
                tipster_data = self._parse_tipster(container)
                if tipster_data:
                    tipsters.append(tipster_data)
                    logger.info(f"Found tipster: {tipster_data['name']}")
        
        except Exception as e:
            logger.error(f"Error parsing tipsters: {e}")
        
        return tipsters
    
    def _parse_tipster(self, tipster_element) -> Optional[Dict[str, Any]]:
        """Parse individual tipster information."""
        try:
            # Extract name
            name_element = tipster_element.find(['h3', 'h4'], class_=re.compile(r'name|title'))
            name = name_element.text.strip() if name_element else "Unknown"
            
            # Extract accuracy
            accuracy_element = tipster_element.find('div', class_=re.compile(r'accuracy|rate'))
            accuracy = 0.0
            if accuracy_element:
                accuracy_text = accuracy_element.text
                accuracy_match = re.search(r'(\d+\.?\d*)%', accuracy_text)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1)) / 100
            
            # Extract ROI
            roi_element = tipster_element.find('div', class_=re.compile(r'roi|profit'))
            roi = 0.0
            if roi_element:
                roi_text = roi_element.text
                roi_match = re.search(r'([+-]?\d+\.?\d*)%', roi_text)
                if roi_match:
                    roi = float(roi_match.group(1))
            
            # Extract tips
            tips_element = tipster_element.find('div', class_=re.compile(r'tips|selections'))
            tips = []
            if tips_element:
                tips = self._parse_tips(tips_element)
            
            # Generate tipster ID
            tipster_id = f"tipster_{name.lower().replace(' ', '_')}"
            
            tipster_data = {
                'tipster_id': tipster_id,
                'name': name,
                'accuracy': accuracy,
                'roi': roi,
                'total_tips': len(tips),
                'tips': tips
            }
            
            return tipster_data
            
        except Exception as e:
            logger.error(f"Error parsing tipster: {e}")
            return None
    
    def _parse_tips(self, tips_element) -> List[Dict[str, Any]]:
        """Parse tips from tipster element."""
        tips = []
        
        try:
            # Look for tip items
            tip_items = tips_element.find_all('div', class_=re.compile(r'tip-item|selection'))
            
            for item in tip_items:
                # Extract race and horse
                race_element = item.find('div', class_=re.compile(r'race|meeting'))
                horse_element = item.find('div', class_=re.compile(r'horse|selection-name'))
                
                if race_element and horse_element:
                    race = race_element.text.strip()
                    horse = horse_element.text.strip()
                    
                    # Extract confidence if available
                    confidence_element = item.find('div', class_=re.compile(r'confidence|rating'))
                    confidence = 0.5  # Default
                    if confidence_element:
                        conf_text = confidence_element.text
                        conf_match = re.search(r'(\d+)/10|(\d+)%', conf_text)
                        if conf_match:
                            if conf_match.group(1):
                                confidence = int(conf_match.group(1)) / 10
                            elif conf_match.group(2):
                                confidence = int(conf_match.group(2)) / 100
                    
                    tip = {
                        'race': race,
                        'horse': horse,
                        'confidence': confidence,
                        'stake': 'win'  # Default stake type
                    }
                    
                    tips.append(tip)
        
        except Exception as e:
            logger.error(f"Error parsing tips: {e}")
        
        return tips
    
    def scrape_form_guide(self, horse_name: str) -> Optional[Dict[str, Any]]:
        """Scrape form guide for a specific horse."""
        logger.info(f"Scraping form guide for {horse_name}")
        
        # Search for horse
        search_url = f"{self.base_url}/search?q={horse_name.replace(' ', '+')}"
        html = self._make_request(search_url)
        
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # Find horse profile link
            profile_link = soup.find('a', href=re.compile(r'/horse/|/profile/'))
            if not profile_link:
                return None
            
            profile_url = self.base_url + profile_link['href']
            profile_html = self._make_request(profile_url)
            
            if not profile_html:
                return None
            
            profile_soup = BeautifulSoup(profile_html, 'html.parser')
            
            # Extract form data
            form_data = {
                'horse_name': horse_name,
                'career_stats': {},
                'recent_form': [],
                'preferred_conditions': {}
            }
            
            # Find stats table
            stats_table = profile_soup.find('table', class_=re.compile(r'stats|career'))
            if stats_table:
                form_data['career_stats'] = self._parse_career_stats(stats_table)
            
            # Find recent runs
            form_table = profile_soup.find('table', class_=re.compile(r'form|runs'))
            if form_table:
                form_data['recent_form'] = self._parse_recent_form(form_table)
            
            # Find going preferences
            going_section = profile_soup.find('div', text=re.compile(r'Going|Ground'))
            if going_section:
                going_stats = going_section.find_next('table')
                if going_stats:
                    form_data['preferred_conditions'] = self._parse_going_stats(going_stats)
            
            return form_data
            
        except Exception as e:
            logger.error(f"Error scraping form guide for {horse_name}: {e}")
            return None
    
    def _parse_career_stats(self, stats_table) -> Dict[str, Any]:
        """Parse career statistics table."""
        stats = {}
        
        try:
            rows = stats_table.find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    stat_name = cells[0].text.strip().lower().replace(' ', '_')
                    stat_value = cells[1].text.strip()
                    
                    # Convert to appropriate type
                    if stat_value.isdigit():
                        stats[stat_name] = int(stat_value)
                    elif re.match(r'^\d+\.?\d*$', stat_value):
                        stats[stat_name] = float(stat_value)
                    else:
                        stats[stat_name] = stat_value
            
        except Exception as e:
            logger.error(f"Error parsing career stats: {e}")
        
        return stats
    
    def _parse_recent_form(self, form_table) -> List[Dict[str, Any]]:
        """Parse recent form/runs table."""
        recent_form = []
        
        try:
            rows = form_table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 8:
                    form_entry = {
                        'date': cells[0].text.strip(),
                        'course': cells[1].text.strip(),
                        'distance': cells[2].text.strip(),
                        'going': cells[3].text.strip(),
                        'position': cells[4].text.strip(),
                        'weight': cells[5].text.strip(),
                        'jockey': cells[6].text.strip(),
                        'odds': cells[7].text.strip()
                    }
                    recent_form.append(form_entry)
            
        except Exception as e:
            logger.error(f"Error parsing recent form: {e}")
        
        return recent_form
    
    def _parse_going_stats(self, going_table) -> Dict[str, Any]:
        """Parse going/ground preferences."""
        going_stats = {}
        
        try:
            rows = going_table.find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    going_type = cells[0].text.strip().lower()
                    stats_text = cells[1].text.strip()
                    
                    # Parse stats like "2-1-3" (wins-seconds-thirds)
                    stats_match = re.match(r'(\d+)-(\d+)-(\d+)', stats_text)
                    if stats_match:
                        going_stats[going_type] = {
                            'wins': int(stats_match.group(1)),
                            'seconds': int(stats_match.group(2)),
                            'thirds': int(stats_match.group(3))
                        }
            
        except Exception as e:
            logger.error(f"Error parsing going stats: {e}")
        
        return going_stats
    
    def save_to_database(self, race_data: Dict[str, Any]) -> bool:
        """Save scraped race data to database."""
        try:
            # Save race
            race = db.save_race(race_data)
            
            if not race:
                logger.error(f"Failed to save race: {race_data['race_id']}")
                return False
            
            # Save runners
            for runner_data in race_data.get('runners', []):
                db.save_runner(race.id, runner_data)
            
            logger.info(f"Successfully saved race {race_data['race_id']} with {len(race_data.get('runners', []))} runners")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return False
    
    def update_results(self, result_data: Dict[str, Any]) -> bool:
        """Update database with race results."""
        try:
            # Find the race in database
            with db.get_session() as session:
                race = session.query(Race).filter_by(race_id=result_data['race_id']).first()
                
                if not race:
                    logger.error(f"Race not found in database: {result_data['race_id']}")
                    return False
                
                # Save result
                result_entry = {
                    'race_id': race.id,
                    'winning_horse_name': result_data['positions'].get(1, 'Unknown'),
                    'second_horse_name': result_data['positions'].get(2, 'Unknown'),
                    'third_horse_name': result_data['positions'].get(3, 'Unknown'),
                    'winning_time': result_data.get('winning_time', 0.0),
                    'dividends': json.dumps(result_data.get('dividends', {}))
                }
                
                db.save_race_result(race.id, result_entry)
                
                # Update race going
                race.going = result_data.get('going', 'Unknown')
                
                logger.info(f"Updated results for race {result_data['race_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating results: {e}")
            return False
    
    def close(self):
        """Clean up resources."""
        if self.use_selenium and hasattr(self, 'driver'):
            self.driver.quit()
            logger.info("Selenium driver closed")
    
    def run_daily_scrape(self):
        """Run complete daily scraping routine."""
        try:
            # Today's race cards
            today = datetime.now()
            races = self.scrape_race_cards(today)
            
            for race in races:
                self.save_to_database(race)
            
            # Yesterday's results
            yesterday = today - timedelta(days=1)
            results = self.scrape_race_results(yesterday)
            
            for result in results:
                self.update_results(result)
            
            # Tipsters
            tipsters = self.scrape_tipsters()
            for tipster in tipsters:
                db.save_tipster(tipster)
            
            logger.info(f"Daily scrape completed: {len(races)} races, {len(results)} results, {len(tipsters)} tipsters")
            
        except Exception as e:
            logger.error(f"Error in daily scrape: {e}")
        finally:
            self.close()
