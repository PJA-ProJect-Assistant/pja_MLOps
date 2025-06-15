import dotenv
import requests
import os
import json
import sys

# 환경변수 로드
dotenv.load_dotenv()
SLACK_URL = os.getenv("SLACK_URL")

class SlackMessage:
    def __init__(self, slack_url=None):
        self.slack_url = slack_url or SLACK_URL
        if not self.slack_url:
            raise ValueError("SLACK_URL이 설정되지 않았습니다. .env 파일을 확인해주세요.")
    
    def send_msg(self, msg):
        """작업 완료 메시지를 Slack으로 전송"""
        message = f"{msg} 작업 완료"
        title = "New Incoming Message :zap:"
        
        slack_data = {
            "username": "NotificationBot",
            "icon_emoji": ":satellite:",
            "attachments": [
                {
                    "color": "#9733EE",
                    "fields": [
                        {
                            "title": title,
                            "value": message,
                            "short": False,
                        }
                    ]
                }
            ]
        }
        
        headers = {'Content-Type': "application/json"}
        
        try:
            response = requests.post(self.slack_url, json=slack_data, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {response.text}")
            return True
        except Exception as e:
            print(f"Slack 메시지 전송 실패: {e}")
            return False

    def send_dataframe_summary(self, df, task_name="데이터 처리"):
        """DataFrame 요약 정보와 함께 메시지 전송"""
        summary = f"{task_name}\n• 행 수: {len(df):,}\n• 열 수: {len(df.columns)}\n• 컬럼: {', '.join(df.columns[:5])}"
        if len(df.columns) > 5:
            summary += f" 외 {len(df.columns)-5}개"
        
        return self.send_msg(summary)

# 편의 함수들
def send_notification(msg):
    """간단한 알림 전송"""
    slack = SlackMessage()
    return slack.send_msg(msg)