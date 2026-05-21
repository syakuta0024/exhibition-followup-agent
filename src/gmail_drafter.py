"""
Gmail API を使ってメールを下書き（Draft）として保存するモジュール。

初回実行時: credentials/credentials.json が必要。ブラウザが開き OAuth 認証を求める。
2回目以降: credentials/token.json が自動利用される。
"""

import base64
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.readonly",
]


class GmailDrafter:
    def __init__(
        self,
        credentials_path: str = "credentials/credentials.json",
        token_path: str = "credentials/token.json",
    ) -> None:
        self._credentials_path = Path(credentials_path)
        self._token_path = Path(token_path)
        self._service = None

    def _get_service(self):
        if self._service is not None:
            return self._service

        creds: Optional[Credentials] = None

        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self._credentials_path.exists():
                    raise FileNotFoundError(
                        f"credentials.json が見つかりません: {self._credentials_path}\n"
                        "Google Cloud Console から OAuth クライアント ID をダウンロードして "
                        "credentials/ フォルダに置いてください。"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(creds.to_json(), encoding="utf-8")

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def create_draft(self, to: str, subject: str, body: str) -> str:
        """
        Gmail の下書きフォルダにメールを1件追加する。

        Parameters
        ----------
        to      : 宛先メールアドレス
        subject : 件名
        body    : 本文（プレーンテキスト）

        Returns
        -------
        str: 作成された draft の ID
        """
        service = self._get_service()

        message = MIMEText(body, "plain", "utf-8")
        message["to"] = to
        message["subject"] = Header(subject, "utf-8")
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        try:
            draft = service.users().drafts().create(
                userId="me",
                body={"message": {"raw": raw}},
            ).execute()
        except HttpError as e:
            raise RuntimeError(f"Gmail API エラー: {e}") from e

        return draft["id"]

    def create_drafts_from_results(self, results: list) -> dict:
        """
        run_generate() の結果リストを受け取り、エラー行を除く全件を下書きに追加する。

        Parameters
        ----------
        results : run_generate() が返す results リスト
                  各要素に email_to / subject / body が含まれていること

        Returns
        -------
        dict: {success: int, errors: int, draft_ids: list[str], error_details: list[str]}
        """
        success = 0
        errors = 0
        draft_ids = []
        error_details = []

        for r in results:
            if r.get("subject") == "ERROR":
                errors += 1
                error_details.append(
                    f"{r.get('visitor_name', '')} ({r.get('company_name', '')}): 生成エラーのためスキップ"
                )
                continue

            to = r.get("email_to", "")
            subject = r.get("subject", "")
            body = r.get("body", "")

            if not to:
                errors += 1
                error_details.append(
                    f"{r.get('visitor_name', '')} ({r.get('company_name', '')}): 宛先メールアドレスが空"
                )
                continue

            try:
                draft_id = self.create_draft(to=to, subject=subject, body=body)
                draft_ids.append(draft_id)
                success += 1
            except Exception as e:
                errors += 1
                error_details.append(
                    f"{r.get('visitor_name', '')} ({r.get('company_name', '')}): {e}"
                )

        return {
            "success": success,
            "errors": errors,
            "draft_ids": draft_ids,
            "error_details": error_details,
        }
