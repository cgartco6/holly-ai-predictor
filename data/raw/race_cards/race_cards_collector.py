import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

class RaceCardsCollector:
    def __init__(self, data_dir="data/raw/race_cards"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def generate_sample_race_cards(self, num_races=20):
        """Generate realistic race card data"""
        np.random.seed(42)
        
        tracks = [
            'Churchill Downs', 'Santa Anita', 'Belmont Park', 
            'Keeneland', 'Gulfstream Park', 'Pimlico', 'Del Mar'
        ]
        
        race_classes = ['Maiden', 'Claiming', 'Allowance', 'Stakes', 'Graded Stakes']
        track_conditions = ['Fast', 'Good', 'Sloppy', 'Wet Fast', 'Muddy']
        distances = [5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0]
        
        all_race_cards = []
        
        for race_id in range(1, num_races + 1):
            race_date = datetime.now() - timedelta(days=np.random.randint(0, 30))
            post_time = race_date.replace(
                hour=np.random.randint(12, 18),
                minute=np.random.choice([0, 15, 30, 45])
            )
            
            race_card = {
                'race_id': f'R{race_id:04d}',
                'race_date': race_date.strftime('%Y-%m-%d'),
                'post_time': post_time.strftime('%H:%M'),
                'track': np.random.choice(tracks),
                'race_name': f'Race {race_id} - {np.random.choice(["Spring", "Summer", "Fall", "Winter"])} Stakes',
                'race_number': race_id % 10 + 1,
                'distance': np.random.choice(distances),
                'surface': np.random.choice(['Dirt', 'Turf']),
                'track_condition': np.random.choice(track_conditions),
                'race_class': np.random.choice(race_classes),
                'purse': np.random.choice([20000, 35000, 50000, 75000, 100000, 250000]),
                'age_restriction': np.random.choice(['2YO', '3YO', '3YO+', '4YO+', 'None']),
                'sex_restriction': np.random.choice(['Colts & Geldings', 'Fillies & Mares', 'Open', 'None']),
                'field_size': np.random.randint(6, 14),
                'horses': []
            }
            
            # Generate horses for this race
            num_horses = race_card['field_size']
            for horse_num in range(1, num_horses + 1):
                horse = self._generate_horse_data(horse_num, race_card)
                race_card['horses'].append(horse)
            
            all_race_cards.append(race_card)
        
        # Save to file
        filename = f"{self.data_dir}/race_cards_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(all_race_cards, f, indent=2)
        
        print(f"Generated {len(all_race_cards)} race cards to {filename}")
        return all_race_cards
    
    def _generate_horse_data(self, horse_num, race_card):
        """Generate individual horse data"""
        horse_names = [
            'Midnight Thunder', 'Silver Streak', 'Golden Crown', 'Red Rocket',
            'Blue Diamond', 'Green Meadow', 'Black Shadow', 'White Knight',
            'Purple Rain', 'Orange Blossom', 'Brown Sugar', 'Grey Ghost'
        ]
        
        jockeys = [
            'John Velazquez', 'Mike Smith', 'Javier Castellano', 
            'Irad Ortiz Jr', 'Flavien Prat', 'Joel Rosario'
        ]
        
        trainers = [
            'Bob Baffert', 'Todd Pletcher', 'Chad Brown',
            'Steve Asmussen', 'Bill Mott', 'Brad Cox'
        ]
        
        horse = {
            'horse_id': f'H{race_card["race_id"][1:]}{horse_num:02d}',
            'program_number': horse_num,
            'horse_name': np.random.choice(horse_names) + f' {horse_num}',
            'jockey': np.random.choice(jockeys),
            'trainer': np.random.choice(trainers),
            'owner': f'{np.random.choice(["Stable", "Farm", "Racing"])} {np.random.choice(["A", "B", "C"])}',
            'age': np.random.randint(2, 8),
            'sex': np.random.choice(['Colt', 'Filly', 'Gelding', 'Mare']),
            'color': np.random.choice(['Bay', 'Chestnut', 'Gray', 'Black', 'Brown']),
            'breeding': {
                'sire': np.random.choice(['Into Mischief', 'Tapit', 'American Pharoah', 'Curlin']),
                'dam': np.random.choice(['Dream', 'Star', 'Moon', 'Sky']) + ' ' + np.random.choice(['Princess', 'Queen', 'Lady', 'Girl'])
            },
            'weight': np.random.randint(1150, 1250),
            'medications': np.random.choice(['Lasix', 'Bute', 'None', 'Both'], p=[0.4, 0.3, 0.2, 0.1]),
            'equipment': np.random.choice(['Blinkers', 'No Blinkers', 'Front Bandages', 'None']),
            'morning_line_odds': round(np.random.uniform(1.5, 20.0), 1),
            'post_position': horse_num,
            'claimed_tag': np.random.choice([0, 15000, 25000, 40000, 50000]),
            'first_time_blinkers': np.random.choice([True, False], p=[0.1, 0.9]),
            'first_time_lasix': np.random.choice([True, False], p=[0.15, 0.85]),
            'workout_data': {
                'last_workout_date': (datetime.now() - timedelta(days=np.random.randint(3, 10))).strftime('%Y-%m-%d'),
                'distance': np.random.choice([3, 4, 5]),
                'time': f'{np.random.randint(35, 38)}.{np.random.randint(0, 60):02d}',
                'track': race_card['track'],
                'condition': np.random.choice(['Fast', 'Good'])
            }
        }
        
        # Add past performance data
        horse['past_performance'] = self._generate_past_performance(horse)
        
        return horse
    
    def _generate_past_performance(self, horse):
        """Generate past performance data for a horse"""
        performances = []
        num_races = np.random.randint(1, 8)
        
        for i in range(num_races):
            race_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
            
            performance = {
                'race_date': race_date.strftime('%Y-%m-%d'),
                'track': np.random.choice(['Churchill Downs', 'Santa Anita', 'Belmont Park']),
                'distance': np.random.choice([5.0, 6.0, 7.0, 8.0, 9.0]),
                'surface': np.random.choice(['Dirt', 'Turf']),
                'track_condition': np.random.choice(['Fast', 'Good', 'Sloppy']),
                'race_class': np.random.choice(['Maiden', 'Claiming', 'Allowance', 'Stakes']),
                'finish_position': np.random.randint(1, 12),
                'beaten_lengths': round(np.random.uniform(0, 15), 1),
                'field_size': np.random.randint(6, 12),
                'odds': round(np.random.uniform(1.5, 25.0), 1),
                'speed_figure': np.random.randint(65, 95),
                'pace_figure': np.random.randint(60, 90),
                'jockey': horse['jockey'] if np.random.random() > 0.3 else np.random.choice(['Different Jockey']),
                'weight_carried': np.random.randint(1140, 1220),
                'comment': np.random.choice([
                    'Won driving', 'Closed well', 'Evenly', 'Faded late',
                    'No factor', 'In hand', 'Rallied', 'Checked early'
                ])
            }
            performances.append(performance)
        
        return performances
    
    def load_race_cards(self, date=None):
        """Load race cards from file"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        filename = f"{self.data_dir}/race_cards_{date}.json"
        
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"No race cards found for {date}, generating sample data...")
            return self.generate_sample_race_cards()

if __name__ == "__main__":
    collector = RaceCardsCollector()
    race_cards = collector.generate_sample_race_cards()
    print(f"Generated {len(race_cards)} race cards")
