"""
Interactive Twitter Influence Prediction Chatbot
================================================
A command-line chatbot that interacts directly with you to predict Twitter influence.
Run this after training models in the Enhanced notebook.

Author: [Your Name]
Course: Master's AI - Machine Learning
Date: November 2025
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class TwitterInfluenceChatbot:
    """Interactive chatbot for Twitter influence prediction"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.user_a_data = {}
        self.user_b_data = {}
        self.conversation_history = []
        
    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*70)
        print("🐦 TWITTER INFLUENCE PREDICTION CHATBOT 🐦")
        print("="*70)
        print("Welcome! I'll help you predict which Twitter user is more influential.")
        print("I'll ask you questions about both users and make a prediction.")
        print("="*70 + "\n")
    
    def load_models(self):
        """Load trained models from pickle files"""
        print("📦 Loading machine learning models...")
        
        # Try to load models
        model_files = {
            'XGBoost': 'best_model_XGBoost.pkl',
            'Gradient_Boosting': 'best_model_Gradient_Boosting.pkl',
            'Random_Forest': 'best_model_Random_Forest.pkl',
        }
        
        loaded_count = 0
        for name, filename in model_files.items():
            try:
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    loaded_count += 1
                    print(f"  ✓ {name} loaded")
            except Exception as e:
                print(f"  ✗ Could not load {name}: {str(e)}")
        
        # Try to load feature names
        try:
            if os.path.exists('feature_names.pkl'):
                with open('feature_names.pkl', 'rb') as f:
                    self.feature_names = pickle.load(f)
                print(f"  ✓ Feature names loaded ({len(self.feature_names)} features)")
        except:
            print("  ⚠ Using default feature names")
            self.feature_names = [
                'A/B_follower_count', 'A/B_following_count', 'A/B_listed_count',
                'A/B_mentions_received', 'A/B_retweets_received', 'A/B_mentions_sent',
                'A/B_retweets_sent', 'A/B_posts', 'A/B_network_feature_1',
                'A/B_network_feature_2', 'A/B_network_feature_3'
            ]
        
        if loaded_count == 0:
            print("\n⚠️  No models loaded! Running in DEMO mode with simple heuristics.")
            print("   To use real ML models, first run the Enhanced notebook to train them.\n")
            self.demo_mode = True
        else:
            print(f"\n✓ Successfully loaded {loaded_count} models!\n")
            self.demo_mode = False
    
    def get_input(self, prompt, input_type='int', min_val=0, max_val=None, allow_skip=False):
        """Get and validate user input"""
        while True:
            try:
                if allow_skip:
                    user_input = input(f"{prompt} (or 'skip' for default): ").strip()
                    if user_input.lower() == 'skip':
                        return None
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using the chatbot! Goodbye!")
                    sys.exit(0)
                
                if input_type == 'int':
                    value = int(user_input)
                elif input_type == 'float':
                    value = float(user_input)
                else:
                    return user_input
                
                if value < min_val:
                    print(f"  ⚠️  Value must be at least {min_val}. Try again.")
                    continue
                
                if max_val is not None and value > max_val:
                    print(f"  ⚠️  Value must be at most {max_val}. Try again.")
                    continue
                
                return value
                
            except ValueError:
                print(f"  ⚠️  Invalid input. Please enter a valid {input_type}.")
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the chatbot! Goodbye!")
                sys.exit(0)
    
    def collect_user_data(self, user_name):
        """Interactive data collection for one user"""
        print(f"\n{'='*70}")
        print(f"📊 Let's collect data for {user_name}")
        print(f"{'='*70}\n")
        
        print(f"I'll ask you 11 questions about {user_name}'s Twitter account.")
        print("Type 'quit' anytime to exit.\n")
        
        data = {}
        
        # Question 1: Followers
        print("📈 Question 1/11: Follower Count")
        data['follower_count'] = self.get_input(
            f"  How many followers does {user_name} have?",
            'int', min_val=0
        )
        
        # Question 2: Following
        print("\n📈 Question 2/11: Following Count")
        data['following_count'] = self.get_input(
            f"  How many accounts does {user_name} follow?",
            'int', min_val=0
        )
        
        # Question 3: Listed
        print("\n📋 Question 3/11: Listed Count")
        print("  (Times this user has been added to Twitter lists)")
        data['listed_count'] = self.get_input(
            f"  Listed count for {user_name}?",
            'int', min_val=0
        )
        
        # Question 4: Mentions Received
        print("\n💬 Question 4/11: Mentions Received")
        print("  (How many times others have mentioned this user)")
        data['mentions_received'] = self.get_input(
            f"  Mentions received by {user_name}?",
            'float', min_val=0
        )
        
        # Question 5: Retweets Received
        print("\n🔄 Question 5/11: Retweets Received")
        print("  (How many times others have retweeted this user)")
        data['retweets_received'] = self.get_input(
            f"  Retweets received by {user_name}?",
            'float', min_val=0
        )
        
        # Question 6: Mentions Sent
        print("\n💬 Question 6/11: Mentions Sent")
        print("  (How many times this user has mentioned others)")
        data['mentions_sent'] = self.get_input(
            f"  Mentions sent by {user_name}?",
            'float', min_val=0
        )
        
        # Question 7: Retweets Sent
        print("\n🔄 Question 7/11: Retweets Sent")
        print("  (How many times this user has retweeted others)")
        data['retweets_sent'] = self.get_input(
            f"  Retweets sent by {user_name}?",
            'float', min_val=0
        )
        
        # Question 8: Posts
        print("\n📝 Question 8/11: Total Posts")
        data['posts'] = self.get_input(
            f"  How many total posts does {user_name} have?",
            'int', min_val=0
        )
        
        # Question 9-11: Network Features
        print("\n🌐 Question 9/11: Network Feature 1")
        print("  (Local follower network metric - average followers of followers)")
        data['network_feature_1'] = self.get_input(
            f"  Network feature 1 for {user_name}?",
            'int', min_val=0
        )
        
        print("\n🌐 Question 10/11: Network Feature 2")
        print("  (Local network metric - engagement density)")
        data['network_feature_2'] = self.get_input(
            f"  Network feature 2 for {user_name}?",
            'float', min_val=0
        )
        
        print("\n🌐 Question 11/11: Network Feature 3")
        print("  (Local network metric - community strength)")
        data['network_feature_3'] = self.get_input(
            f"  Network feature 3 for {user_name}?",
            'float', min_val=0
        )
        
        print(f"\n✅ All data collected for {user_name}!")
        
        return data
    
    def display_summary(self):
        """Display summary of collected data"""
        print("\n" + "="*70)
        print("📊 DATA SUMMARY")
        print("="*70 + "\n")
        
        print("👤 USER A:")
        print(f"  Followers: {self.user_a_data['follower_count']:,}")
        print(f"  Following: {self.user_a_data['following_count']:,}")
        print(f"  Posts: {self.user_a_data['posts']:,}")
        print(f"  Listed: {self.user_a_data['listed_count']:,}")
        print(f"  Engagement: {self.user_a_data['mentions_received']:.1f} mentions, {self.user_a_data['retweets_received']:.1f} retweets")
        
        print("\n👤 USER B:")
        print(f"  Followers: {self.user_b_data['follower_count']:,}")
        print(f"  Following: {self.user_b_data['following_count']:,}")
        print(f"  Posts: {self.user_b_data['posts']:,}")
        print(f"  Listed: {self.user_b_data['listed_count']:,}")
        print(f"  Engagement: {self.user_b_data['mentions_received']:.1f} mentions, {self.user_b_data['retweets_received']:.1f} retweets")
        
        print("\n" + "="*70)
    
    def calculate_ratios(self):
        """Calculate A/B ratio features"""
        epsilon = 1e-10
        ratios = {}
        
        for key in self.user_a_data.keys():
            ratio_key = f'A/B_{key}'
            ratios[ratio_key] = self.user_a_data[key] / (self.user_b_data[key] + epsilon)
        
        return ratios
    
    def predict_with_models(self, ratios):
        """Make predictions using loaded models"""
        results = {}
        
        # Convert ratios to DataFrame
        input_df = pd.DataFrame([ratios])
        
        # Ensure columns match training data
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[self.feature_names]
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                results[model_name] = {
                    'prediction': 'User B' if prediction == 1 else 'User A',
                    'confidence': max(probabilities) * 100,
                    'prob_a': probabilities[0] * 100,
                    'prob_b': probabilities[1] * 100
                }
            except Exception as e:
                print(f"  ⚠️  Error with {model_name}: {str(e)}")
        
        return results
    
    def predict_demo_mode(self, ratios):
        """Simple heuristic prediction for demo mode"""
        follower_ratio = ratios['A/B_follower_count']
        listed_ratio = ratios.get('A/B_listed_count', 1.0)
        engagement_ratio = (ratios.get('A/B_mentions_received', 1.0) + 
                          ratios.get('A/B_retweets_received', 1.0)) / 2
        
        # Weighted score
        score = (follower_ratio * 0.4 + listed_ratio * 0.3 + engagement_ratio * 0.3)
        
        if score > 1.5:
            prediction = 'User A'
            confidence = min(70 + (score - 1.5) * 10, 95)
        elif score < 0.67:
            prediction = 'User B'
            confidence = min(70 + (1/score - 1.5) * 10, 95)
        else:
            if score >= 1.0:
                prediction = 'User A'
                confidence = 50 + (score - 1.0) * 40
            else:
                prediction = 'User B'
                confidence = 50 + (1.0 - score) * 40
        
        prob_a = confidence if prediction == 'User A' else 100 - confidence
        prob_b = 100 - prob_a
        
        return {
            'Demo Heuristic': {
                'prediction': prediction,
                'confidence': confidence,
                'prob_a': prob_a,
                'prob_b': prob_b
            }
        }
    
    def display_prediction(self, results):
        """Display prediction results beautifully"""
        print("\n" + "="*70)
        print("🎯 PREDICTION RESULTS")
        print("="*70 + "\n")
        
        # Get consensus prediction
        predictions = [r['prediction'] for r in results.values()]
        winner = max(set(predictions), key=predictions.count)
        consensus = predictions.count(winner) / len(predictions) * 100
        
        print(f"🏆 CONSENSUS WINNER: {winner}")
        print(f"   Agreement: {consensus:.0f}% of models agree")
        print(f"\n{'─'*70}\n")
        
        # Individual model results
        print("📊 Individual Model Predictions:\n")
        
        for model_name, result in results.items():
            pred = result['prediction']
            conf = result['confidence']
            
            # Visual confidence bar
            bar_length = int(conf / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            print(f"  {model_name:20s} → {pred:7s} | {bar} {conf:.1f}%")
        
        print(f"\n{'─'*70}\n")
        
        # Probability breakdown
        avg_prob_a = np.mean([r['prob_a'] for r in results.values()])
        avg_prob_b = np.mean([r['prob_b'] for r in results.values()])
        
        print("📈 Average Probabilities:")
        print(f"  User A: {avg_prob_a:.1f}%")
        print(f"  User B: {avg_prob_b:.1f}%")
        
        # Visual probability bars
        print("\n  User A: [", end="")
        print('█' * int(avg_prob_a / 5), end="")
        print('░' * (20 - int(avg_prob_a / 5)), end="")
        print(f"] {avg_prob_a:.1f}%")
        
        print("  User B: [", end="")
        print('█' * int(avg_prob_b / 5), end="")
        print('░' * (20 - int(avg_prob_b / 5)), end="")
        print(f"] {avg_prob_b:.1f}%")
        
        print("\n" + "="*70)
    
    def display_analysis(self, ratios):
        """Display feature analysis"""
        print("\n" + "="*70)
        print("📊 FEATURE ANALYSIS")
        print("="*70 + "\n")
        
        print("Top 5 Most Important Features (from research):\n")
        
        important_features = [
            ('A/B_follower_count', 'Follower Ratio', 15.6),
            ('A/B_network_feature_3', 'Network Strength', 14.7),
            ('A/B_listed_count', 'Listed Ratio', 13.5),
            ('A/B_network_feature_2', 'Network Density', 8.9),
            ('A/B_retweets_received', 'Retweet Ratio', 8.4)
        ]
        
        for feature_key, feature_name, importance in important_features:
            if feature_key in ratios:
                ratio = ratios[feature_key]
                advantage = "User A" if ratio > 1 else "User B" if ratio < 1 else "Equal"
                
                print(f"  {feature_name:20s} | Ratio: {ratio:6.2f} | Advantage: {advantage:7s} | Importance: {importance:.1f}%")
        
        print("\n" + "="*70)
    
    def ask_continue(self):
        """Ask if user wants to make another prediction"""
        print("\n" + "="*70)
        response = input("\n🔄 Would you like to make another prediction? (yes/no): ").strip().lower()
        return response in ['yes', 'y', 'yeah', 'yep', 'sure']
    
    def quick_mode(self):
        """Quick mode with example data"""
        print("\n" + "="*70)
        print("⚡ QUICK MODE - Using Example Data")
        print("="*70 + "\n")
        
        self.user_a_data = {
            'follower_count': 50000,
            'following_count': 1000,
            'listed_count': 500,
            'mentions_received': 100.0,
            'retweets_received': 50.0,
            'mentions_sent': 20.0,
            'retweets_sent': 10.0,
            'posts': 5000,
            'network_feature_1': 200,
            'network_feature_2': 50.0,
            'network_feature_3': 1000.0
        }
        
        self.user_b_data = {
            'follower_count': 20000,
            'following_count': 2000,
            'listed_count': 200,
            'mentions_received': 50.0,
            'retweets_received': 30.0,
            'mentions_sent': 40.0,
            'retweets_sent': 20.0,
            'posts': 3000,
            'network_feature_1': 100,
            'network_feature_2': 30.0,
            'network_feature_3': 500.0
        }
        
        print("Using example data:")
        print("  User A: 50K followers, 5K posts, high engagement")
        print("  User B: 20K followers, 3K posts, moderate engagement\n")
    
    def run(self):
        """Main chatbot loop"""
        self.display_banner()
        self.load_models()
        
        while True:
            # Ask for mode
            print("\nChoose mode:")
            print("  1. Full Mode - Enter all data yourself")
            print("  2. Quick Mode - Use example data")
            print("  3. Exit")
            
            mode = self.get_input("\nYour choice (1/2/3)", 'int', min_val=1, max_val=3)
            
            if mode == 3:
                print("\n👋 Thanks for using the chatbot! Goodbye!")
                break
            elif mode == 2:
                self.quick_mode()
            else:
                # Collect data for both users
                self.user_a_data = self.collect_user_data("User A")
                self.user_b_data = self.collect_user_data("User B")
            
            # Display summary
            self.display_summary()
            
            # Calculate ratios
            print("\n🧮 Calculating feature ratios...")
            ratios = self.calculate_ratios()
            print("✓ Feature engineering complete!")
            
            # Make prediction
            print("\n🤖 Making predictions with ML models...")
            
            if self.demo_mode:
                results = self.predict_demo_mode(ratios)
                print("✓ Prediction complete! (using simple heuristics)")
            else:
                results = self.predict_with_models(ratios)
                print(f"✓ Predictions complete! ({len(results)} models)")
            
            # Display results
            self.display_prediction(results)
            self.display_analysis(ratios)
            
            # Save to history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_a': self.user_a_data,
                'user_b': self.user_b_data,
                'results': results
            })
            
            # Ask to continue
            if not self.ask_continue():
                print("\n👋 Thanks for using the Twitter Influence Prediction Chatbot!")
                print(f"📊 Total predictions made: {len(self.conversation_history)}")
                print("="*70 + "\n")
                break

# Main execution
if __name__ == "__main__":
    chatbot = TwitterInfluenceChatbot()
    chatbot.run()
