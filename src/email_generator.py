"""
フォローアップメール生成モジュール

OpenAI GPT (gpt-5.4-nano) を使い、商談確度・関心製品・顧客属性に応じた
パーソナライズされたフォローアップメールを生成する。
"""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import Config
from src.utils import setup_logger, parse_interested_products

logger = setup_logger(__name__)

# ---------------------------------------------------------------
# 商談確度別のメール方針定義（業界ベストプラクティス準拠）
# ---------------------------------------------------------------
RANK_POLICY: Dict[str, Dict[str, str]] = {
    "A": {
        "label": "ホットリード - 即商談",
        "tone": "積極的・具体的・パーソナル",
        "instruction": """
商談確度が最高（A）のホットリードです。以下のルールに従ってメールを作成してください:
- 件名に【御礼】と展示会名を含める
- 展示会での具体的な会話内容を冒頭で振り返る（「○○についてお話しさせていただきましたが」）
- メモに記載された顧客の具体的な課題・数値を引用する
- 関心製品の導入効果を定量的な数値（%、万円）で示す
- 類似業種の導入事例を1つ具体的に紹介する
- CRM過去履歴がある場合は「以前よりお取引の検討を頂戴しており」等の文脈を入れる
- 「来週中に一度お時間を頂戴できませんか」と具体的なデモ・商談日程を打診する
- 2〜3の候補日を提示する形のCTAにする
- メール全体は400〜600字程度
""",
    },
    "B": {
        "label": "ウォームリード - フォロー",
        "tone": "丁寧・提案型・情報提供",
        "instruction": """
商談確度B（ウォームリード）です。以下のルールに従ってメールを作成してください:
- 件名に【御礼】と製品名を含める
- 展示会でのお礼から始め、関心を持っていただいた製品に触れる
- 関心製品の詳細資料（PDF）を添付する旨を案内する
- 顧客の課題に対する解決提案を1〜2点具体的に述べる
- 導入事例の定量効果を1つ紹介する
- CTA: 「まずはオンラインで30分ほどご説明の機会をいただけませんか」
- 社内検討用の資料が必要であれば用意する旨を添える
- メール全体は300〜500字程度
""",
    },
    "C": {
        "label": "ウォームリード - 情報提供",
        "tone": "控えめ・情報提供型・プレッシャーなし",
        "instruction": """
商談確度C（ライトフォロー）です。以下のルールに従ってメールを作成してください:
- 件名に【御礼】を含め、シンプルに
- 展示会でのご来場のお礼を述べる
- 関心を持った製品の概要資料のURLまたはPDFを案内する
- 「ご参考までにお送りいたします」というスタンス
- プレッシャーをかけず、押し売り感を出さない
- CTA: 「ご関心がございましたらお気軽にお問い合わせください」程度
- セミナーや展示会の次回開催案内があれば一言添える
- メール全体は200〜350字程度
""",
    },
    "D": {
        "label": "コールドリード - 御礼",
        "tone": "シンプル・短い・親しみやすい",
        "instruction": """
商談確度D（コールドリード）です。以下のルールに従ってメールを作成してください:
- 件名は「【御礼】○○展にご来場いただきありがとうございました」のシンプルな形式
- ご来場のお礼を簡潔に述べる
- コスト面に配慮があるお客様なので、中小企業向けスタータープランや無料トライアルがあることを一言紹介する
- 積極的な商談提案はしない
- CTA: 「ご不明点がございましたらお気軽にご連絡ください」のみ
- メール全体は150〜250字程度の短いもの
""",
    },
    "E": {
        "label": "コールドリード - 最小接触",
        "tone": "最小限・定型的",
        "instruction": """
商談確度E（最小限接触）です。以下のルールに従ってメールを作成してください:
- 件名は「【御礼】○○展ご来場の御礼」のみ
- ご来場のお礼を2〜3行で述べるだけ
- 製品カタログのご案内を一文添える（具体URLは記載しない／設定済みURLがある場合のみそれを使用する）
- 商談提案やCTAは一切不要
- 押しつけがましくならないよう注意
- メール全体は100〜150字以内の非常に短いもの
""",
    },
}

# ---------------------------------------------------------------
# 出力フォーマットの指定（LLMへの指示）
# 【CTA】は「メール本文内のCTA」ではなく「営業担当向けの内部アクション指示」
# ---------------------------------------------------------------
def _build_output_format_instruction(
    sender_company: str = "",
    sender_name: str = "",
    product_urls: Optional[Dict[str, str]] = None,
) -> str:
    """出力フォーマット指示を組み立てる（sender_company / sender_name / product_urls を展開して LLM に渡す）"""
    company = sender_company or "弊社"
    name = sender_name or "●●"

    configured_urls = {k: v for k, v in (product_urls or {}).items() if v.strip()}
    if configured_urls:
        url_lines = "\n".join(f"  - {k}: {v}" for k, v in configured_urls.items())
        url_rule = (
            f"※ 本文にURLを記載する場合は、以下の設定済みURLのみを使用してください:\n"
            f"{url_lines}\n"
            "  上記以外のURL（推測URL・example.com等のサンプルドメイン）は絶対に使用しないでください。"
        )
    else:
        url_rule = (
            "※ 本文にURLを含めないでください。\n"
            "  製品URLは未設定です。推測URL・example.com・example.org等のサンプルドメインも絶対に使用しないでください。"
        )

    return f"""
以下のフォーマットで厳密に出力してください。

【件名】
（メールの件名。【御礼】を含めること）

【本文】
（完全なメール文面。以下の構成で書くこと：
1. 宛名（会社名・部署・役職・氏名 様）
2. 挨拶（お世話になっております。{company} 営業部の{name}でございます。）
   ※ 送信元は必ず「{company}」です。顧客の会社名を送信元として使わないこと。
3. 展示会来場のお礼
4. 展示会での会話の振り返り（確度A/Bの場合のみ）
5. 製品情報・提案（確度に応じて深さを変える）
6. 次のアクション提案（確度に応じて）
7. 締めの挨拶
8. 署名
※ 上記の番号（1〜8）は構成ガイドです。本文には番号を出力しないでください。
※ 件名・本文に「L001」「L007」のような内部管理コード・参照IDを含めないでください。
※ メールでは「ご関心製品」欄に記載された製品のみに言及してください。
  そこに含まれない製品を新たに勧めないでください。
※ 「ご関心製品」に複数製品がある場合、メール本文では主要1〜2製品に絞って
  言及してください。3つ以上すべてに触れる必要はありません。
{url_rule}）

【CTA】
（営業担当者が次にとるべき社内向けアクション指示を1〜2文で記載。
メール本文の内容ではなく、営業が次に実行すべきことを具体的に書く。
例: 「3営業日以内に電話でフォロー。デモ環境（EdgeGuard + Sorani）を事前に準備しておく」）
"""


class EmailGenerator:
    """
    フォローアップメールを生成するクラス。

    OpenAI GPT を使って、リード情報とベクトルDB検索結果をもとに
    件名・本文・CTAを含む完全なメール文面を生成する。
    """

    def __init__(self, llm_model: str = Config.LLM_MODEL, temperature: float = 0.7):
        """
        Parameters
        ----------
        llm_model : str
            使用するOpenAIモデル名（デフォルト: gpt-5.4-nano）
        temperature : float
            生成温度（デフォルト: 0.7 ── 自然な文体のため）
        """
        # ChatOpenAI の初期化
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=Config.OPENAI_API_KEY,
        )
        logger.info(f"EmailGenerator 初期化完了 (model={llm_model}, temperature={temperature})")

    def generate(
        self,
        lead: Dict[str, Any],
        tech_context: str = "",
        crm_context: str = "",
        crm_structured: Optional[Dict[str, Any]] = None,
        exhibition_info: Optional[Dict[str, str]] = None,
        web_context: str = "",
        sender_company: str = "",
        sender_name: str = "",
        audio_context: str = "",
        product_urls: Optional[Dict[str, str]] = None,
        schedule_context: str = "",
        product_card_context: str = "",
    ) -> Dict[str, str]:
        """
        フォローアップメールを生成する。

        Parameters
        ----------
        lead : Dict[str, Any]
            リード情報（leads.csvの1行分を辞書化したもの）
        tech_context : str
            ベクトルDB検索で取得した技術資料テキスト
        crm_context : str
            ベクトルDB検索で取得したCRM記録テキスト（テキスト形式）
        crm_structured : Dict[str, Any], optional
            CRM CSV から取得した構造化商談データ。
            存在する場合はテキスト形式のcrm_contextより優先して使用される。

        Returns
        -------
        Dict[str, str]
            生成されたメール {'subject': ..., 'body': ..., 'cta': ...}
        """
        rank = str(lead.get("lead_rank", "C")).upper()
        policy = RANK_POLICY.get(rank, RANK_POLICY["C"])

        logger.info(
            f"{lead.get('visitor_name')}さん ({lead.get('company_name')}) の"
            f"メール生成中... [ランク{rank}: {policy['label']}]"
        )

        # プロンプト組み立て
        system_prompt = self._build_system_prompt(sender_company=sender_company, sender_name=sender_name)
        _common = dict(
            lead=lead, policy=policy, tech_context=tech_context, crm_context=crm_context,
            crm_structured=crm_structured, exhibition_info=exhibition_info,
            web_context=web_context, audio_context=audio_context,
            sender_company=sender_company, sender_name=sender_name, product_urls=product_urls,
            product_card_context=product_card_context,
        )
        if schedule_context.strip():
            human_prompt = self._build_human_prompt_with_schedule(
                **_common, schedule_context=schedule_context
            )
        else:
            human_prompt = self._build_human_prompt_info_only(**_common)

        # LLM呼び出し
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            response = self.llm.invoke(messages)
            raw_text = response.content

        except Exception as e:
            logger.error(f"LLM呼び出しエラー: {e}")
            raise RuntimeError(f"メール生成に失敗しました: {e}") from e

        # レスポンスをパース
        result = self._parse_llm_response(raw_text)
        logger.info(f"  → 生成完了: 件名「{result['subject']}」")
        return result

    def _build_system_prompt(self, sender_company: str = "", sender_name: str = "") -> str:
        """システムプロンプトを返す"""
        company = sender_company or "弊社"
        if sender_name:
            name_instruction = (
                f"メールの名乗り（「お世話になっております。{company} 営業部の{sender_name}でございます」等）"
                f"と署名には必ず「{sender_name}」を使用してください。"
                "架空の名前や「●●」などのプレースホルダーを使わないでください。"
            )
            signature = f"{company} 営業部\n{sender_name}"
        else:
            name_instruction = (
                "メールの名乗りと署名には必ず「●●」とプレースホルダーを使用してください。"
                "架空の名前や具体的な人名を絶対に生成しないでください。"
            )
            signature = f"{company} 営業部\n●●"
        return (
            f"あなたは{company}の営業担当者です。\n"
            f"{name_instruction}\n\n"
            "展示会にご来場いただいたお客様に対して、丁寧かつ効果的なフォローアップメールを"
            "日本語ビジネスメール形式で作成してください。\n\n"
            "自社の製品・サービス情報は提供されるコンテキスト（技術資料・過去商談記録）を参照し、"
            "実際に提供できる内容のみをメールに含めてください。\n\n"
            "署名は以下を使用してください:\n"
            "---\n"
            f"{signature}\n"
            "---\n"
            "件名・本文に「L001」「L007」のような内部管理IDを含めないでください。"
        )

    def _assemble_context_sections(
        self,
        lead: Dict[str, Any],
        tech_context: str,
        crm_context: str,
        crm_structured: Optional[Dict[str, Any]],
        exhibition_info: Optional[Dict[str, str]],
        web_context: str,
        audio_context: str,
        schedule_context: str = "",
        product_card_context: str = "",
    ) -> str:
        """共通のコンテキストセクションを組み立てて返す。
        schedule_context を渡さない呼び出しでは面談候補日は含まれない。"""

        # ── CRMセクション: 構造化データ優先 ──────────────────────
        crm_section = ""
        if crm_structured and any(v.strip() for v in crm_structured.values() if isinstance(v, str)):
            match_method = crm_structured.get("match_method", "")
            method_label = "メール一致" if match_method == "email" else "会社名マッチ"
            lines = [f"## 過去のCRM情報（HubSpot連携 / 紐付け: {method_label}）"]
            if crm_structured.get("lifecycle_stage"):
                lines.append(f"- ライフサイクルステージ: {crm_structured['lifecycle_stage']}")
            if crm_structured.get("lead_status"):
                lines.append(f"- リードステータス: {crm_structured['lead_status']}")
            if crm_structured.get("last_activity_date"):
                lines.append(f"- 最終接触日: {crm_structured['last_activity_date']}")
            if crm_structured.get("create_date"):
                lines.append(f"- 初回登録日: {crm_structured['create_date']}")
            if crm_structured.get("original_source"):
                lines.append(f"- 獲得経路: {crm_structured['original_source']}")
            if crm_structured.get("contact_owner"):
                lines.append(f"- 担当者: {crm_structured['contact_owner']}")
            crm_section = "\n".join(lines) + "\n"
        elif crm_context.strip():
            crm_section = f"## 過去のCRM商談履歴（参考）\n{crm_context}\n"

        # ── 製品カードセクション（human-verified、技術資料より優先） ───────
        product_card_section = ""
        if product_card_context.strip():
            product_card_section = (
                "## ★製品情報（人間確認済み）\n"
                "※ 以下は確認済みの正確な製品情報です。\n"
                "  技術資料(RAG)と矛盾する場合はこちらを優先してください。\n"
                f"{product_card_context}\n"
            )

        # ── 技術資料セクション ───────────────────────────────────
        tech_section = ""
        if tech_context.strip():
            tech_section = f"## 関心製品の技術資料（参考）\n{tech_context}\n"

        # ── 追加情報セクション（独自アンケート等）───────────────────
        extra_info_lines = []
        for key, val in lead.items():
            if key.startswith("extra_") and str(val).strip():
                label = key[len("extra_"):]
                extra_info_lines.append(f"- {label}: {val}")
        extra_section = ""
        if extra_info_lines:
            extra_section = "## 追加情報（アンケート・独自項目）\n" + "\n".join(extra_info_lines) + "\n"

        # ── 展示会情報セクション ─────────────────────────────────
        _ex = exhibition_info or {}
        _ex_name = _ex.get("exhibition_name") or "展示会"
        _ex_date = _ex.get("exhibition_date") or ""
        _ex_venue = _ex.get("exhibition_venue") or ""
        _ex_lines = ["## 展示会情報", f"- 展示会名: {_ex_name}"]
        if _ex_date:
            _ex_lines.append(f"- 開催日: {_ex_date}")
        if _ex_venue:
            _ex_lines.append(f"- 会場: {_ex_venue}")
        exhibition_section = "\n".join(_ex_lines) + "\n"

        # ── Web検索結果セクション ────────────────────────────────
        web_section = ""
        if web_context.strip():
            web_section = (
                f"## 顧客企業の最新情報（Web検索結果）\n{web_context}\n\n"
                "※ 上記の最新情報が関連する場合、メール内で自然に触れてください。"
                "不自然になる場合は無視して構いません。\n"
            )

        # ── 音声コンテキストセクション（最優先）──────────────────────
        audio_section = ""
        if audio_context.strip():
            audio_section = (
                "## ★最優先情報（録音音声より）\n"
                "※ 以下の情報を最も重視してメールを作成してください。\n"
                f"{audio_context}\n"
            )

        # ── 面談候補日セクション（schedule_context が渡された場合のみ）──
        schedule_section = ""
        if schedule_context.strip():
            schedule_section = f"## 面談候補日\n{schedule_context}\n"

        joined = "\n".join(
            s for s in [audio_section, exhibition_section, schedule_section,
                        extra_section, crm_section, product_card_section, tech_section, web_section]
            if s.strip()
        )
        return ("\n" + joined) if joined else ""

    def _build_human_prompt_with_schedule(
        self,
        lead: Dict[str, Any],
        policy: Dict[str, str],
        tech_context: str,
        crm_context: str,
        crm_structured: Optional[Dict[str, Any]] = None,
        exhibition_info: Optional[Dict[str, str]] = None,
        web_context: str = "",
        audio_context: str = "",
        sender_company: str = "",
        sender_name: str = "",
        product_urls: Optional[Dict[str, str]] = None,
        schedule_context: str = "",
        product_card_context: str = "",
    ) -> str:
        """候補日3つを提示する A/B 向けプロンプト構築。"""
        products = parse_interested_products(str(lead.get("interested_products", "")))
        products_str = "、".join(products) if products else "（未記載）"
        context_sections = self._assemble_context_sections(
            lead, tech_context, crm_context, crm_structured, exhibition_info,
            web_context, audio_context, schedule_context=schedule_context,
            product_card_context=product_card_context,
        )
        schedule_structure = (
            "\n## メール構成（この顧客は候補日提示型として扱う）\n"
            "1. お礼挨拶\n"
            "2. 当日の会話に触れる（memo から引用）\n"
            "3. 関心製品（主要1〜2）の具体的な提案\n"
            "4. 「以下の候補日のいずれかでお打ち合わせいかがでしょうか」と提示\n"
            "5. ## 面談候補日 セクションの3候補をそのまま組み込む（日付・曜日・時間帯）\n"
            "※ 番号は構成ガイドです。本文には番号を出力しないでください。\n"
        )
        return f"""以下のお客様情報をもとにフォローアップメールを作成してください。

## お客様情報
- 氏名: {lead.get('visitor_name', '')} 様
- 会社名: {lead.get('company_name', '')}
- 部署・役職: {lead.get('department', '')} / {lead.get('job_title', '')}
- メールアドレス: {lead.get('email', '')}
- 来場日: {lead.get('visit_date', '')}
- ご関心製品: {products_str}
- 今後のご要望: {lead.get('future_requests', '（なし）')}
- 展示会での会話メモ: {lead.get('memo', '（なし）')}
{context_sections}
## メール作成方針
- ランク: {lead.get('lead_rank', 'C')} ({policy['label']})
- トーン: {policy['tone']}
- 指示: {policy['instruction']}
{schedule_structure}
{_build_output_format_instruction(sender_company, sender_name, product_urls)}"""

    def _build_human_prompt_info_only(
        self,
        lead: Dict[str, Any],
        policy: Dict[str, str],
        tech_context: str,
        crm_context: str,
        crm_structured: Optional[Dict[str, Any]] = None,
        exhibition_info: Optional[Dict[str, str]] = None,
        web_context: str = "",
        audio_context: str = "",
        sender_company: str = "",
        sender_name: str = "",
        product_urls: Optional[Dict[str, str]] = None,
        product_card_context: str = "",
    ) -> str:
        """候補日提示なし・情報提供型（C/D/E）向けプロンプト構築。"""
        products = parse_interested_products(str(lead.get("interested_products", "")))
        products_str = "、".join(products) if products else "（未記載）"
        # schedule_context を渡さない → schedule_section は空
        context_sections = self._assemble_context_sections(
            lead, tech_context, crm_context, crm_structured, exhibition_info,
            web_context, audio_context, product_card_context=product_card_context,
        )
        info_structure = (
            "\n## メール構成（この顧客は情報提供型として扱う）\n"
            "1. お礼挨拶\n"
            "2. 当日の会話に触れる（memo から引用）\n"
            "3. 関心製品（主要1〜2）の簡単な情報提供\n"
            "4. 「ご興味があればお気軽にお問い合わせください」のような押し付けない結び\n"
            "5. 候補日は提示しない（候補日提示型ではない）\n"
            "※ 番号は構成ガイドです。本文には番号を出力しないでください。\n"
        )
        return f"""以下のお客様情報をもとにフォローアップメールを作成してください。

## お客様情報
- 氏名: {lead.get('visitor_name', '')} 様
- 会社名: {lead.get('company_name', '')}
- 部署・役職: {lead.get('department', '')} / {lead.get('job_title', '')}
- メールアドレス: {lead.get('email', '')}
- 来場日: {lead.get('visit_date', '')}
- ご関心製品: {products_str}
- 今後のご要望: {lead.get('future_requests', '（なし）')}
- 展示会での会話メモ: {lead.get('memo', '（なし）')}
{context_sections}
## メール作成方針
- ランク: {lead.get('lead_rank', 'C')} ({policy['label']})
- トーン: {policy['tone']}
- 指示: {policy['instruction']}
{info_structure}
{_build_output_format_instruction(sender_company, sender_name, product_urls)}"""

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """
        LLMのレスポンスをパースして件名・本文・CTAに分割する。

        Parameters
        ----------
        response : str
            LLMの生成テキスト

        Returns
        -------
        Dict[str, str]
            {'subject': ..., 'body': ..., 'cta': ...}
        """
        result = {"subject": "", "body": "", "cta": ""}

        # 【件名】【本文】【CTA】を区切りに分割
        sections = {"subject": "【件名】", "body": "【本文】", "cta": "【CTA】"}

        for key, marker in sections.items():
            if marker in response:
                # マーカーの次から次のマーカーまでを取得
                start = response.index(marker) + len(marker)
                # 次のマーカーを探す
                next_markers = [m for m in sections.values() if m != marker and m in response[start:]]
                if next_markers:
                    # 最も近い次のマーカーの手前まで
                    end = min(response.index(m, start) for m in next_markers if response.find(m, start) != -1)
                    result[key] = response[start:end].strip()
                else:
                    result[key] = response[start:].strip()

        # パース失敗時のフォールバック
        if not result["subject"] and not result["body"]:
            logger.warning("レスポンスのパースに失敗しました。全文をbodyに格納します。")
            result["body"] = response.strip()

        return result

    def batch_generate(self, leads_df, vectordb_manager) -> List[Dict]:
        """
        複数リードに対してメールを一括生成する。

        Parameters
        ----------
        leads_df : pd.DataFrame
            全リードデータ
        vectordb_manager : VectorDBManager
            ベクトルDB検索インスタンス（tech_docs / crm の検索に使用）

        Returns
        -------
        List[Dict]
            各リードに対する生成メール結果のリスト
        """
        results = []
        total = len(leads_df)

        for i, row in leads_df.iterrows():
            lead = row.to_dict()
            logger.info(f"[{i+1}/{total}] {lead.get('visitor_name')} ({lead.get('company_name')}) 処理中...")

            try:
                # 技術資料検索
                products = parse_interested_products(str(lead.get("interested_products", "")))
                tech_query = " ".join(products) if products else ""
                tech_results = vectordb_manager.search_tech_docs(tech_query, top_k=3) if tech_query else []
                tech_context = "\n\n".join(r["text"][:400] for r in tech_results)

                # CRM検索
                crm_results = vectordb_manager.search_crm(lead.get("company_name", ""), top_k=2)
                crm_context = "\n\n".join(r["text"][:400] for r in crm_results)

                # メール生成
                email = self.generate(lead, tech_context=tech_context, crm_context=crm_context)

                results.append({
                    "lead_id": lead.get("lead_id"),
                    "visitor_name": lead.get("visitor_name"),
                    "company_name": lead.get("company_name"),
                    "lead_rank": lead.get("lead_rank"),
                    "email_to": lead.get("email"),
                    "subject": email["subject"],
                    "body": email["body"],
                    "cta": email["cta"],
                })

            except Exception as e:
                logger.error(f"  エラー ({lead.get('visitor_name')}): {e}")
                results.append({
                    "lead_id": lead.get("lead_id"),
                    "visitor_name": lead.get("visitor_name"),
                    "company_name": lead.get("company_name"),
                    "lead_rank": lead.get("lead_rank"),
                    "email_to": lead.get("email"),
                    "subject": "ERROR",
                    "body": str(e),
                    "cta": "",
                })

        logger.info(f"一括生成完了: {len(results)}件")
        return results
