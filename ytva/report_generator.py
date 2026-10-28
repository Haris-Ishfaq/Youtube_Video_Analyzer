"""
Report Generator Module
Generates detailed reports of all analysis tasks with viral suggestions
"""

import json
import csv
from datetime import datetime
import os

class ReportGenerator:
    def __init__(self):
        self.report_dir = 'reports'
        os.makedirs(self.report_dir, exist_ok=True)
    
    def _calculate_viral_score(self, summary):
        """Calculate viral potential score based on analysis data"""
        toxic_count = summary.get('toxic_count', 0)
        spam_count = summary.get('spam_count', 0)
        bot_count = summary.get('bot_count', 0)
        normal_count = summary.get('normal_count', 0)
        total_comments = summary.get('total_comments', 1)
        
        viral_score = 70  # Base score
        
        # Factor 1: Toxicity reduces score
        toxic_penalty = min(30, toxic_count * 2)
        viral_score -= toxic_penalty
        
        # Factor 2: Spam reduces score
        spam_penalty = min(20, spam_count * 1.5)
        viral_score -= spam_penalty
        
        # Factor 3: Bot activity reduces authenticity
        bot_penalty = min(15, bot_count * 0.5)
        viral_score -= bot_penalty
        
        # Factor 4: Normal comments increase score
        normal_bonus = min(25, normal_count / 2)
        viral_score += normal_bonus
        
        # Factor 5: Human engagement bonus
        human_count = total_comments - bot_count
        human_bonus = min(15, human_count / 3)
        viral_score += human_bonus
        
        return max(0, min(100, round(viral_score)))
    
    def _generate_suggestions(self, summary, comments):
        """Generate improvement suggestions based on analysis data"""
        suggestions = []
        toxic_count = summary.get('toxic_count', 0)
        spam_count = summary.get('spam_count', 0)
        bot_count = summary.get('bot_count', 0)
        normal_count = summary.get('normal_count', 0)
        total_comments = summary.get('total_comments', 1)
        
        # Check for toxicity issues
        if toxic_count > 5:
            suggestions.append({
                'title': 'High Toxicity Detected',
                'description': f'{toxic_count} toxic comments were found. Toxic comments can drive away genuine viewers and reduce engagement.',
                'priority': 'urgent',
                'action': 'Enable comment moderation or reply respectfully to defuse tensions.'
            })
        elif toxic_count > 2:
            suggestions.append({
                'title': 'Moderate Toxicity Detected',
                'description': f'{toxic_count} toxic comments found. Consider addressing these to maintain a healthy community.',
                'priority': 'high',
                'action': 'Review and moderate toxic comments to improve community atmosphere.'
            })
        
        # Check for spam issues
        if spam_count > 10:
            suggestions.append({
                'title': 'Spam Comments are Hurting Engagement',
                'description': f'{spam_count} spam comments detected. Excessive spam makes your comment section look unprofessional.',
                'priority': 'urgent',
                'action': 'Use YouTube\'s spam filters or add a moderation team to clean up spam.'
            })
        elif spam_count > 5:
            suggestions.append({
                'title': 'Spam Comments Detected',
                'description': f'{spam_count} spam comments found. Consider implementing stronger spam filtering.',
                'priority': 'high',
                'action': 'Enable comment moderation and report spam accounts.'
            })
        
        # Check for bot activity
        if bot_count > 5:
            suggestions.append({
                'title': 'Bot Activity Detected',
                'description': f'{bot_count} bot-like comments detected. Bots can artificially inflate engagement metrics.',
                'priority': 'high',
                'action': 'Consider using CAPTCHA or stricter comment verification for new accounts.'
            })
        elif bot_count > 2:
            suggestions.append({
                'title': 'Suspicious Bot Activity',
                'description': f'{bot_count} bot-like comments detected. Monitor for unusual comment patterns.',
                'priority': 'medium',
                'action': 'Review suspicious accounts and report bot activity to YouTube.'
            })
        
        # Check normal comment percentage
        normal_percentage = (normal_count / total_comments) * 100 if total_comments > 0 else 0
        if normal_percentage < 50:
            suggestions.append({
                'title': 'Low Positive Engagement',
                'description': f'Only {normal_percentage:.1f}% of comments are positive/normal. This indicates audience dissatisfaction.',
                'priority': 'urgent',
                'action': 'Review your content strategy and address audience concerns raised in comments.'
            })
        elif normal_percentage < 70:
            suggestions.append({
                'title': 'Moderate Positive Engagement',
                'description': f'{normal_percentage:.1f}% of comments are positive. Room for improvement.',
                'priority': 'medium',
                'action': 'Engage with positive comments and create more content that resonates with your audience.'
            })
        
        # Extract top themes from comments
        themes = self._extract_top_themes(comments)
        if themes:
            suggestions.append({
                'title': 'What Your Audience Loves',
                'description': f'Your audience is most engaged with: {", ".join(themes[:3])}. Create more content around these topics!',
                'priority': 'medium',
                'action': 'Plan your next video around the most discussed themes in comments.'
            })
        
        # Add general improvement suggestions
        suggestions.append({
            'title': 'Consistency is Key',
            'description': 'Videos with consistent upload schedules tend to perform better. Your comment analysis can help identify what your audience wants.',
            'priority': 'medium',
            'action': 'Analyze which topics get the most positive engagement and create more content in that niche.'
        })
        
        suggestions.append({
            'title': 'Build a Community, Not Just Views',
            'description': 'Respond to top comments, ask follow-up questions, and create a sense of belonging.',
            'priority': 'low',
            'action': 'Reply to at least 5 comments in your next video within the first hour of posting.'
        })
        
        suggestions.append({
            'title': 'Optimize Your Video Title and Thumbnail',
            'description': 'Titles and thumbnails are the number one factor for click-through rate.',
            'priority': 'high',
            'action': 'Test 2-3 different thumbnails and titles for your next video.'
        })
        
        return suggestions
    
    def _extract_top_themes(self, comments):
        """Extract common themes from comments"""
        themes = []
        keywords = ['tutorial', 'review', 'funny', 'music', 'gaming', 'tech', 
                   'education', 'entertainment', 'news', 'sports', 'camera', 
                   'iphone', 'android', 'python', 'coding', 'how to']
        
        for comment in comments[:50]:  # Check first 50 comments
            text = (comment.get('text', '')).lower()
            for keyword in keywords:
                if keyword in text and keyword not in themes:
                    themes.append(keyword)
                    if len(themes) >= 5:
                        break
            if len(themes) >= 5:
                break
        
        return themes
    
    def generate_html_report(self, analysis_data):
        """
        Generate a beautiful HTML report with viral suggestions
        """
        video_id = analysis_data.get('video_id', 'Unknown')
        video_title = analysis_data.get('video_title', 'YouTube Video')
        summary = analysis_data.get('summary', {})
        comments = analysis_data.get('comments', [])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate additional stats
        toxic_count = summary.get('toxic_count', 0)
        spam_count = summary.get('spam_count', 0)
        normal_count = summary.get('normal_count', 0)
        bot_count = summary.get('bot_count', 0)
        total_comments = summary.get('total_comments', 0)
        human_count = total_comments - bot_count
        
        normal_percentage = summary.get('normal_pct', 0)
        spam_percentage = summary.get('spam_pct', 0)
        toxic_percentage = summary.get('toxic_pct', 0)
        bot_percentage = summary.get('bot_pct', 0)
        human_percentage = summary.get('human_pct', 100 - bot_percentage)
        
        # Calculate viral score
        viral_score = self._calculate_viral_score(summary)
        
        # Determine viral level
        if viral_score >= 70:
            viral_level = "HIGH VIRAL POTENTIAL"
            viral_color = "#10b981"
        elif viral_score >= 50:
            viral_level = "MEDIUM VIRAL POTENTIAL"
            viral_color = "#f59e0b"
        elif viral_score >= 30:
            viral_level = "LOW VIRAL POTENTIAL"
            viral_color = "#ef4444"
        else:
            viral_level = "VERY LOW VIRAL POTENTIAL"
            viral_color = "#6b7280"
        
        # Generate suggestions
        suggestions = self._generate_suggestions(summary, comments)
        
        # Extract top comments
        top_toxic = [c for c in comments if c.get('label') == 'toxic'][:10]
        top_spam = [c for c in comments if c.get('label') == 'spam'][:10]
        top_bot = [c for c in comments if c.get('is_bot')][:10]
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Video Analysis Report - {video_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e2e8f0;
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(17, 24, 39, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #818cf8;
            margin: 25px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #2d3748;
        }}
        h3 {{
            margin: 15px 0 10px 0;
            color: #c4b5fd;
        }}
        .header-info {{
            background: rgba(99, 102, 241, 0.1);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 25px;
            border-left: 4px solid #6366f1;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            background: rgba(255,255,255,0.08);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #818cf8;
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 5px;
        }}
        .progress-bar {{
            height: 8px;
            background: #2d3748;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .fill-normal {{ background: #10b981; }}
        .fill-spam {{ background: #f59e0b; }}
        .fill-toxic {{ background: #ef4444; }}
        .fill-bot {{ background: #f59e0b; }}
        
        /* Viral Score Styles */
        .viral-score-container {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 25px;
            text-align: center;
        }}
        .viral-score-circle {{
            width: 140px;
            height: 140px;
            margin: 0 auto 15px;
        }}
        .viral-score-text {{
            font-size: 2rem;
            font-weight: 800;
        }}
        .viral-label {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 8px;
        }}
        
        /* Suggestion Styles */
        .suggestion-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .suggestion-card {{
            background: rgba(99,102,241,0.08);
            border-radius: 12px;
            padding: 15px;
            border-left: 3px solid;
            transition: all 0.3s;
        }}
        .suggestion-card:hover {{
            transform: translateY(-3px);
            background: rgba(99,102,241,0.12);
        }}
        .suggestion-card.urgent {{ border-left-color: #ef4444; }}
        .suggestion-card.high {{ border-left-color: #f59e0b; }}
        .suggestion-card.medium {{ border-left-color: #10b981; }}
        .suggestion-card.low {{ border-left-color: #6366f1; }}
        
        .suggestion-title {{
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 1rem;
        }}
        .suggestion-desc {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 10px;
        }}
        .priority-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        .priority-urgent {{ background: #ef444420; color: #ef4444; }}
        .priority-high {{ background: #f59e0b20; color: #f59e0b; }}
        .priority-medium {{ background: #10b98120; color: #10b981; }}
        .priority-low {{ background: #6366f120; color: #818cf8; }}
        
        /* Action Button */
        .action-btn {{
            background: linear-gradient(135deg, #f59e0b, #d97706);
            border: none;
            border-radius: 8px;
            color: white;
            padding: 6px 12px;
            font-size: 0.7rem;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 10px;
        }}
        .action-btn:hover {{
            transform: translateY(-2px);
            opacity: 0.9;
        }}
        
        /* Checklist Styles */
        .improvement-item {{
            background: rgba(255,255,255,0.03);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        .improvement-check {{ color: #10b981; margin-right: 10px; }}
        .improvement-cross {{ color: #ef4444; margin-right: 10px; }}
        
        /* Table Styles */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.85rem;
        }}
        th {{
            background: rgba(99, 102, 241, 0.2);
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }}
        .badge-normal {{ background: #10b98120; color: #10b981; border: 1px solid #10b98140; }}
        .badge-spam {{ background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }}
        .badge-toxic {{ background: #ef444420; color: #ef4444; border: 1px solid #ef444440; }}
        .badge-bot {{ background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }}
        .badge-human {{ background: #10b98120; color: #10b981; border: 1px solid #10b98140; }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #2d3748;
            font-size: 0.8rem;
            color: #64748b;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>YouTube Video Analysis Report</h1>
    <div class="header-info">
        <p><strong>Video Title:</strong> {self._escape_html(video_title)}</p>
        <p><strong>Video ID:</strong> {video_id}</p>
        <p><strong>Analysis Date:</strong> {timestamp}</p>
        <p><strong>Total Comments Analyzed:</strong> {total_comments}</p>
    </div>

    <!-- Viral Potential Score -->
    <div class="viral-score-container">
        <h3>Viral Potential Score</h3>
        <div class="viral-score-circle">
            <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="62" fill="none" stroke="#2d3748" stroke-width="8"/>
                <circle cx="70" cy="70" r="62" fill="none" stroke="{viral_color}" stroke-width="8"
                        stroke-dasharray="{viral_score * 3.89} 389" stroke-dashoffset="0"
                        transform="rotate(-90 70 70)"/>
                <text x="70" y="80" text-anchor="middle" fill="{viral_color}" font-size="28" font-weight="bold">{viral_score}%</text>
                <text x="70" y="95" text-anchor="middle" fill="#94a3b8" font-size="10">Viral Score</text>
            </svg>
        </div>
        <div class="viral-score-text" style="color: {viral_color}">{viral_level}</div>
        <div class="viral-label">Based on toxicity, spam, bot detection, and audience engagement metrics</div>
    </div>

    <!-- Summary Stats -->
    <h2>Overall Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total_comments}</div>
            <div class="stat-label">Total Comments</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #10b981;">{normal_percentage}%</div>
            <div class="stat-label">Normal Comments ({normal_count})</div>
            <div class="progress-bar"><div class="progress-fill fill-normal" style="width: {normal_percentage}%"></div></div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #f59e0b;">{spam_percentage}%</div>
            <div class="stat-label">Spam Comments ({spam_count})</div>
            <div class="progress-bar"><div class="progress-fill fill-spam" style="width: {spam_percentage}%"></div></div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #ef4444;">{toxic_percentage}%</div>
            <div class="stat-label">Toxic Comments ({toxic_count})</div>
            <div class="progress-bar"><div class="progress-fill fill-toxic" style="width: {toxic_percentage}%"></div></div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #f59e0b;">{bot_percentage}%</div>
            <div class="stat-label">Suspected Bots ({bot_count})</div>
            <div class="progress-bar"><div class="progress-fill fill-bot" style="width: {bot_percentage}%"></div></div>
        </div>
    </div>

    <!-- Viral Growth Suggestions -->
    <h2>Viral Growth Suggestions and Improvements</h2>
    <div class="suggestion-grid">
'''

        for sug in suggestions[:6]:  # Show top 6 suggestions
            priority_class = sug['priority']
            html_content += f'''
        <div class="suggestion-card {priority_class}">
            <div class="suggestion-title">{self._escape_html(sug['title'])}</div>
            <div class="suggestion-desc">{self._escape_html(sug['description'])}</div>
            <div>
                <span class="priority-badge priority-{priority_class}">{priority_class.upper()} PRIORITY</span>
            </div>
            <div class="action-btn" onclick="alert('Action: {self._escape_html(sug['action'])}')">
                Take Action
            </div>
        </div>
'''

        html_content += f'''
    </div>

    <!-- Quick Wins Checklist -->
    <h2>Quick Wins Checklist</h2>
    <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 15px; margin-bottom: 25px;">
        <div class="improvement-item">
            <span class="improvement-{'check' if spam_count < 5 else 'cross'}">{'✅' if spam_count < 5 else '❌'}</span>
            Clean up spam comments ({spam_count} detected)
        </div>
        <div class="improvement-item">
            <span class="improvement-{'check' if toxic_count < 3 else 'cross'}">{'✅' if toxic_count < 3 else '❌'}</span>
            Address toxic comments ({toxic_count} detected)
        </div>
        <div class="improvement-item">
            <span class="improvement-{'check' if normal_percentage > 70 else 'cross'}">{'✅' if normal_percentage > 70 else '❌'}</span>
            Positive engagement rate: {normal_percentage}% {('(Good!)' if normal_percentage > 70 else '(Needs improvement)')}
        </div>
        <div class="improvement-item">
            <span class="improvement-{'check' if bot_count == 0 else 'cross'}">{'✅' if bot_count == 0 else '❌'}</span>
            Bot activity: {bot_count} bot-like comments detected
        </div>
        <div class="improvement-item">
            <span>📊</span>
            Next video suggestion: Create content based on the most discussed themes from your audience
        </div>
    </div>

    <!-- Toxic Comments Section -->
    <h2>Top Toxic Comments</h2>
'''

        if top_toxic:
            html_content += '''
    <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr><th>#</th><th>Author</th><th>Comment</th><th>Confidence</th></tr>
            </thead>
            <tbody>
'''
            for i, c in enumerate(top_toxic, 1):
                html_content += f'''
                <tr>
                    <td>{i}</td>
                    <td>{self._escape_html(c.get('author', 'Unknown'))}</td>
                    <td>"{self._escape_html(c.get('text', '')[:80])}..."</td>
                    <td><span class="badge badge-toxic">{c.get('confidence', 0)}%</span></td>
                </tr>
'''
            html_content += '''
            </tbody>
        </table>
    </div>
'''
        else:
            html_content += '<p>No toxic comments detected in this dataset.</p>'

        # Spam Comments Section
        html_content += '''
    <h2>Top Spam Comments</h2>
'''
        if top_spam:
            html_content += '''
    <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr><th>#</th><th>Author</th><th>Comment</th><th>Confidence</th></tr>
            </thead>
            <tbody>
'''
            for i, c in enumerate(top_spam, 1):
                html_content += f'''
                <tr>
                    <td>{i}</td>
                    <td>{self._escape_html(c.get('author', 'Unknown'))}</td>
                    <td>"{self._escape_html(c.get('text', '')[:80])}..."</td>
                    <td><span class="badge badge-spam">{c.get('confidence', 0)}%</span></td>
                </tr>
'''
            html_content += '''
            </tbody>
        </table>
    </div>
'''
        else:
            html_content += '<p>No spam comments detected in this dataset.</p>'

        # Bot Comments Section
        html_content += '''
    <h2>Suspicious Bot Comments</h2>
'''
        if top_bot:
            html_content += '''
    <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr><th>#</th><th>Author</th><th>Comment</th><th>Bot Probability</th></tr>
            </thead>
            <tbody>
'''
            for i, c in enumerate(top_bot, 1):
                html_content += f'''
                <tr>
                    <td>{i}</td>
                    <td>{self._escape_html(c.get('author', 'Unknown'))}</td>
                    <td>"{self._escape_html(c.get('text', '')[:80])}..."</td>
                    <td><span class="badge badge-bot">{c.get('bot_probability', 0)}%</span></td>
                </tr>
'''
            html_content += '''
            </tbody>
        </table>
    </div>
'''
        else:
            html_content += '<p>No bot activity detected in this dataset.</p>'

        # All Comments Table
        html_content += f'''
    <h2>All Comments (First 50)</h2>
    <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Author</th>
                    <th>Comment</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Bot Status</th>
                </tr>
            </thead>
            <tbody>
'''

        for i, comment in enumerate(comments[:50], 1):
            label = comment.get('label', 'normal')
            confidence = comment.get('confidence', 0)
            is_bot = comment.get('is_bot', False)
            author = comment.get('author', 'Unknown')[:30]
            text = comment.get('text', '')[:80]
            
            label_class = f'badge-{label}'
            badge_text = label.upper()
            
            bot_badge = '<span class="badge badge-bot">BOT</span>' if is_bot else '<span class="badge badge-human">HUMAN</span>'
            
            html_content += f'''
                <tr>
                    <td>{i}</td>
                    <td>{self._escape_html(author)}</td>
                    <td>"{self._escape_html(text)}..."</td>
                    <td><span class="badge {label_class}">{badge_text}</span></td>
                    <td>{confidence}%</td>
                    <td>{bot_badge}</td>
                </tr>
'''

        html_content += f'''
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>Report generated by YouTube Video Analyzer (YT-VA)</p>
        <p>Use these insights to optimize your content strategy and increase viral potential</p>
        <p>Department of Software Engineering, UAJK | Final Year Project 2025-26</p>
    </div>
</div>
<script>
    function showAction(action) {{
        alert('Action Item: ' + action);
    }}
</script>
</body>
</html>
'''
        
        # Save HTML file
        filename = f'report_{video_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        filepath = os.path.join(self.report_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def generate_csv_report(self, analysis_data):
        """
        Generate CSV report for spreadsheet analysis
        """
        video_id = analysis_data.get('video_id', 'Unknown')
        comments = analysis_data.get('comments', [])
        
        filename = f'report_{video_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        filepath = os.path.join(self.report_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['author', 'comment', 'category', 'confidence', 'is_bot', 'bot_probability', 'likes', 'published_at']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for comment in comments:
                writer.writerow({
                    'author': comment.get('author', ''),
                    'comment': comment.get('text', ''),
                    'category': comment.get('label', 'normal'),
                    'confidence': comment.get('confidence', 0),
                    'is_bot': 'Yes' if comment.get('is_bot') else 'No',
                    'bot_probability': comment.get('bot_probability', 0),
                    'likes': comment.get('likes', 0),
                    'published_at': comment.get('published_at', '')
                })
        
        return filepath
    
    def generate_json_report(self, analysis_data):
        """
        Generate JSON report for API integration
        """
        video_id = analysis_data.get('video_id', 'Unknown')
        summary = analysis_data.get('summary', {})
        comments = analysis_data.get('comments', [])
        
        # Add viral suggestions to JSON
        suggestions = self._generate_suggestions(summary, comments)
        viral_score = self._calculate_viral_score(summary)
        
        filename = f'report_{video_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        filepath = os.path.join(self.report_dir, filename)
        
        report_data = {
            'metadata': {
                'video_id': video_id,
                'video_title': analysis_data.get('video_title', 'Unknown'),
                'generated_at': datetime.now().isoformat(),
                'total_comments': len(comments)
            },
            'summary': summary,
            'viral_analysis': {
                'viral_score': viral_score,
                'suggestions': suggestions
            },
            'comments': comments
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _escape_html(self, text):
        """Escape HTML special characters"""
        if not text:
            return ''
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')