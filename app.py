import os
import re
import warnings
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or os.getenv("GOOGLE_API_KEY") or ""
os.environ["GEMINI_API_KEY"] = api_key
os.environ.setdefault("GOOGLE_API_KEY", api_key)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import Flow, start, listen, router, or_, and_
from crewai.tools import tool

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
llm = LLM(model=MODEL_NAME, api_key=api_key)


def ask(prompt: str) -> str:
    """Single direct LLM call returning plain text."""
    return str(llm.call([{"role": "user", "content": prompt}])).strip()


@tool("DuckDuckGoSearch")
def duckduckgo_search(query: str) -> str:
    """Searches DuckDuckGo and returns concise factual snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        return "\n\n".join(f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results)
    except Exception as e:
        return f"Search unavailable, answer from your own knowledge. ({e})"


def make_agent(role, goal, backstory, tools=None):
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools or [],
                 llm=llm, verbose=False, allow_delegation=False)


class Query(BaseModel):
    input: str


class Answer(BaseModel):
    output: str


app = FastAPI(title="CrewAI 10-Projects")


@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "online", "routes": [f"/agent{i}" for i in range(1, 11)], "docs": "/docs"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


def safe(fn, text: str) -> Answer:
    """Always return {"output": ...}, even on failure, so the format never breaks."""
    try:
        text = (text or "").strip()
        if not text:
            return Answer(output="Error: 'input' must be a non-empty string.")
        return Answer(output=str(fn(text)))
    except Exception as e:
        return Answer(output=f"Error: {type(e).__name__}: {e}")


def split_on(text: str, marker: str):
    """Split 'main text ... marker: extra' into (main, extra)."""
    m = re.search(rf"{marker}\s*:", text, flags=re.I)
    if not m:
        return text.strip(), ""
    return text[:m.start()].strip(), text[m.end():].strip()


# =====================================================================
# 1. Blog Post Crew (sequential)  -- input: topic
# =====================================================================
def run_agent1(topic: str) -> str:
    researcher = make_agent("Topic Researcher", "Find key factual developments on {topic}",
                            "You gather accurate insights for technical writing.", [duckduckgo_search])
    writer = make_agent("Blog Writer", "Draft a clean 500-word blog post based on research",
                        "You write engaging, educational markdown posts.")
    t1 = Task(description="Research the topic '{topic}' and list 4-5 core insights.",
              expected_output="Bullet list of core facts.", agent=researcher)
    t2 = Task(description="Write a 500-word markdown blog post on '{topic}' using the research.",
              expected_output="Full markdown article.", agent=writer, context=[t1])
    crew = Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.sequential, verbose=False)
    return crew.kickoff(inputs={"topic": topic})


@app.post("/agent1", response_model=Answer, tags=["CrewAI"])
def agent1(q: Query):
    return safe(run_agent1, q.input)


# =====================================================================
# 2. Competitor Analysis Crew  -- input: competitors (e.g. "Notion vs Coda")
# =====================================================================
def run_agent2(competitors: str) -> str:
    feat = make_agent("Feature Analyst", "Extract features for {competitors}",
                      "Extracts product specifications and values.", [duckduckgo_search])
    price = make_agent("Pricing Specialist", "Analyze pricing tiers for {competitors}",
                       "Monetization and pricing expert.", [duckduckgo_search])
    strat = make_agent("Chief Strategist", "Combine features and pricing into a SWOT table",
                       "Synthesizes competitor battlecards.")
    t1 = Task(description="Analyze features of {competitors}.", expected_output="Features list.", agent=feat)
    t2 = Task(description="Analyze pricing of {competitors}.", expected_output="Pricing table.", agent=price)
    t3 = Task(description="Produce a markdown SWOT table comparing {competitors}.",
              expected_output="Complete SWOT table.", agent=strat, context=[t1, t2])
    crew = Crew(agents=[feat, price, strat], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)
    return crew.kickoff(inputs={"competitors": competitors})


@app.post("/agent2", response_model=Answer, tags=["CrewAI"])
def agent2(q: Query):
    return safe(run_agent2, q.input)


# =====================================================================
# 3. Meeting Notes Crew (parallel extraction)  -- input: transcript
# =====================================================================
def run_agent3(transcript: str) -> str:
    summarizer = make_agent("Summarizer", "Create 5-bullet summary", "Minutes recorder.")
    actions = make_agent("Action Tracker", "Extract action items with owners", "Task tracker.")
    fmt = make_agent("Formatter", "Assemble final meeting document", "Executive secretary.")
    t_sum = Task(description="Summarize this meeting transcript in 5 bullets:\n{transcript}",
                 expected_output="5 bullets.", agent=summarizer, async_execution=True)
    t_act = Task(description="Extract a table of Action | Owner | Due date from this transcript:\n{transcript}",
                 expected_output="Action table.", agent=actions, async_execution=True)
    t_fmt = Task(description="Assemble the summary and actions into markdown with headings '## Summary' and '## Action Items'.",
                 expected_output="Formatted meeting notes.", agent=fmt, context=[t_sum, t_act])
    crew = Crew(agents=[summarizer, actions, fmt], tasks=[t_sum, t_act, t_fmt], verbose=False)
    return crew.kickoff(inputs={"transcript": transcript})


@app.post("/agent3", response_model=Answer, tags=["CrewAI"])
def agent3(q: Query):
    return safe(run_agent3, q.input)


# =====================================================================
# 4. Data Report Crew (structured output)  -- input: CSV text
# =====================================================================
class SalesMetrics(BaseModel):
    total_revenue: float
    total_units_sold: int
    top_selling_product: str
    key_drivers: List[str]


def run_agent4(csv_text: str) -> str:
    analyst = make_agent("Data Analyst", "Extract metrics from raw sales data", "Statistician.")
    writer = make_agent("Report Writer", "Write strategic business insights", "BI author.")

    def build(structured: bool):
        t1 = Task(description="Read this sales data and compute total revenue, total units sold, "
                              "top selling product and key drivers:\n{csv_content}",
                  expected_output="Metrics matching the SalesMetrics schema.", agent=analyst,
                  **({"output_pydantic": SalesMetrics} if structured else {}))
        t2 = Task(description="Explain the top 3 commercial findings based on the calculated metrics.",
                  expected_output="3 strategic paragraphs.", agent=writer, context=[t1])
        return Crew(agents=[analyst, writer], tasks=[t1, t2], process=Process.sequential, verbose=False)

    try:
        result = build(True).kickoff(inputs={"csv_content": csv_text})
    except Exception:  # structured parsing failed -> fall back to plain text
        result = build(False).kickoff(inputs={"csv_content": csv_text})
    metrics = ""
    try:
        m = result.tasks_output[0].pydantic
        if m:
            metrics = f"Metrics: {m.model_dump_json()}\n\n"
    except Exception:
        pass
    return metrics + str(result)


@app.post("/agent4", response_model=Answer, tags=["CrewAI"])
def agent4(q: Query):
    return safe(run_agent4, q.input)


# =====================================================================
# 5. Customer Support Flow (router + or_)  -- input: customer message
# =====================================================================
class SupportState(BaseModel):
    message: str = ""
    category: str = ""
    priority: str = "normal"
    resolution: str = ""


class CustomerSupportFlow(Flow[SupportState]):
    @start()
    def receive_ticket(self):
        return self.state.message

    @listen(receive_ticket)
    def categorize_ticket(self, body):
        self.state.category = ask(
            f"Categorize into exactly one word: 'billing', 'access', or 'other'. Message: {body}").lower()
        return self.state.category

    @router(categorize_ticket)
    def route_by_category(self):
        if "billing" in self.state.category:
            return "billing"
        if "access" in self.state.category:
            return "access"
        return "other"

    @listen("billing")
    def handle_billing(self):
        self.state.priority = "high"
        return "Refund voucher generated and payment ledger synchronized."

    @listen("access")
    def handle_access(self):
        self.state.priority = "high"
        return "Temporary password reset link dispatched to verified email."

    @listen("other")
    def handle_other(self):
        self.state.priority = "normal"
        return "Added to standard queue with relevant FAQ self-serve links."

    @listen(or_(handle_billing, handle_access, handle_other))
    def resolve_ticket(self, action):
        self.state.resolution = action
        return (f"RESOLVED | Category: {self.state.category.upper()} | "
                f"Priority: {self.state.priority.upper()} | Action: {action}")


def run_agent5(text: str) -> str:
    flow = CustomerSupportFlow()
    flow.state.message = text
    return flow.kickoff()


@app.post("/agent5", response_model=Answer, tags=["CrewAI"])
def agent5(q: Query):
    return safe(run_agent5, q.input)


# =====================================================================
# 6. Resume Screening Flow  -- input: resume text (optionally "Requirements: ...")
# =====================================================================
class ResumeState(BaseModel):
    name: str = "Candidate"
    resume_text: str = ""
    requirements: str = "General professional suitability for the role."
    score: int = 0
    decision: str = ""
    outcome: str = ""


class ResumeScreeningFlow(Flow[ResumeState]):
    @start()
    def load_resume(self):
        return self.state.resume_text

    @listen(load_resume)
    def score_resume(self, text):
        raw = ask(f"Evaluate the candidate against the requirements: {self.state.requirements}\n"
                  f"Resume: {text}\nReturn ONLY an integer score from 0 to 10.")
        digits = re.findall(r"\d+", raw)
        self.state.score = min(int(digits[0]), 10) if digits else 5
        return self.state.score

    @router(score_resume)
    def check_score(self):
        return "shortlist" if self.state.score >= 7 else "reject"

    @listen("shortlist")
    def run_summary_crew(self):
        self.state.decision = "SHORTLISTED"
        ev = make_agent("Recruiter", "List candidate strengths and gaps", "Talent Lead.")
        task = Task(description=f"Summarize candidate strengths and gaps for {self.state.name}:\n{self.state.resume_text}",
                    expected_output="Candidate dossier.", agent=ev)
        self.state.outcome = str(Crew(agents=[ev], tasks=[task], verbose=False).kickoff())
        return self.state.outcome

    @listen("reject")
    def draft_rejection(self):
        self.state.decision = "REJECTED"
        self.state.outcome = (f"Dear {self.state.name}, thank you for your application. "
                              "We are moving forward with other candidates.")
        return self.state.outcome

    @listen(or_(run_summary_crew, draft_rejection))
    def save_results(self, text):
        return f"Decision: {self.state.decision} ({self.state.score}/10)\n{text}"


def run_agent6(text: str) -> str:
    resume, reqs = split_on(text, "requirements")
    flow = ResumeScreeningFlow()
    m = re.search(r"name\s*:\s*([^\n,;]+)", resume, flags=re.I)
    if m:
        flow.state.name = m.group(1).strip()
    flow.state.resume_text = resume
    if reqs:
        flow.state.requirements = reqs
    return flow.kickoff()


@app.post("/agent6", response_model=Answer, tags=["CrewAI"])
def agent6(q: Query):
    return safe(run_agent6, q.input)


# =====================================================================
# 7. Bug Triage Flow  -- input: issue title/body
# =====================================================================
class IssueState(BaseModel):
    text: str = ""
    issue_type: str = ""
    reply: str = ""


class BugTriageFlow(Flow[IssueState]):
    @start()
    def read_issue(self):
        return self.state.text

    @listen(read_issue)
    def classify_issue(self, text):
        self.state.issue_type = ask(
            f"Classify this issue into 'bug', 'feature', or 'question'. Return ONLY that word.\n\n{text}").lower()
        return self.state.issue_type

    @router(classify_issue)
    def route_type(self):
        if "bug" in self.state.issue_type:
            return "bug"
        if "feature" in self.state.issue_type:
            return "feature"
        return "question"

    def _crew(self, role, goal, backstory, desc, expected):
        a = make_agent(role, goal, backstory)
        t = Task(description=desc, expected_output=expected, agent=a)
        return str(Crew(agents=[a], tasks=[t], verbose=False).kickoff())

    @listen("bug")
    def run_bug_crew(self):
        return self._crew("QA Lead", "Identify severity and repro steps", "Diagnostic engineer.",
                          f"Analyze this bug report:\n{self.state.text}", "Severity (P1-P4) and repro steps.")

    @listen("feature")
    def run_feature_crew(self):
        return self._crew("Product Manager", "Write User Story and Acceptance Criteria", "Product lead.",
                          f"Draft a user story for this request:\n{self.state.text}", "Formatted user story.")

    @listen("question")
    def run_answer_crew(self):
        return self._crew("Support Dev", "Provide step-by-step solution", "Dev specialist.",
                          f"Answer this question:\n{self.state.text}", "Step-by-step answer.")

    @listen(or_(run_bug_crew, run_feature_crew, run_answer_crew))
    def post_reply(self, reply):
        self.state.reply = reply
        return f"TRIAGE: [{self.state.issue_type.upper()}]\n{reply}"


def run_agent7(text: str) -> str:
    flow = BugTriageFlow()
    flow.state.text = text
    return flow.kickoff()


@app.post("/agent7", response_model=Answer, tags=["CrewAI"])
def agent7(q: Query):
    return safe(run_agent7, q.input)


# =====================================================================
# 8. Content Repurposing Flow (parallel + and_)  -- input: article text
# =====================================================================
class RepurposeState(BaseModel):
    article: str = ""
    points: str = ""
    linkedin: str = ""
    twitter: str = ""
    newsletter: str = ""


def _one_task(role, goal, backstory, desc, expected):
    a = make_agent(role, goal, backstory)
    t = Task(description=desc, expected_output=expected, agent=a)
    return str(Crew(agents=[a], tasks=[t], verbose=False).kickoff())


class ContentRepurposeFlow(Flow[RepurposeState]):
    @start()
    def load_article(self):
        return self.state.article

    @listen(load_article)
    def extract_points(self, article):
        self.state.points = ask(f"Extract 3 core takeaways from this text:\n{article}")
        return self.state.points

    @listen(extract_points)
    def make_linkedin(self, points):
        self.state.linkedin = _one_task("LinkedIn Writer", "Write 150-word post", "B2B copywriter.",
                                        f"Draft a LinkedIn post based on:\n{points}",
                                        "LinkedIn post with hook and hashtags.")
        return self.state.linkedin

    @listen(extract_points)
    def make_twitter(self, points):
        self.state.twitter = _one_task("Twitter Creator", "Write a 3-tweet thread", "Tech writer.",
                                       f"Write a 3-tweet thread (1/3, 2/3, 3/3) based on:\n{points}",
                                       "Numbered tweet thread.")
        return self.state.twitter

    @listen(extract_points)
    def make_newsletter(self, points):
        self.state.newsletter = _one_task("Newsletter Editor", "Write a quick newsletter blurb", "Email curator.",
                                          f"Write a newsletter blurb with intro and 3 takeaways based on:\n{points}",
                                          "Newsletter section.")
        return self.state.newsletter

    @listen(and_(make_linkedin, make_twitter, make_newsletter))
    def publish_pack(self):
        return (f"## LinkedIn Post\n{self.state.linkedin}\n\n"
                f"## Twitter Thread\n{self.state.twitter}\n\n"
                f"## Newsletter Digest\n{self.state.newsletter}")


def run_agent8(text: str) -> str:
    flow = ContentRepurposeFlow()
    flow.state.article = text
    return flow.kickoff()


@app.post("/agent8", response_model=Answer, tags=["CrewAI"])
def agent8(q: Query):
    return safe(run_agent8, q.input)


# =====================================================================
# 9. Lead Qualification Flow  -- input: "Company name, https://website"
# =====================================================================
class LeadState(BaseModel):
    company: str = ""
    website: str = ""
    enrichment: str = ""
    tier: str = ""


class LeadQualificationFlow(Flow[LeadState]):
    @start()
    def capture_lead(self):
        return f"{self.state.company} ({self.state.website})"

    @listen(capture_lead)
    def enrich_lead(self, info):
        self.state.enrichment = _one_task("B2B Researcher", "Find business domain and scale", "Lead researcher.",
                                          f"Summarize business model and scale for: {info}", "3-point company brief.")
        return self.state.enrichment

    @router(enrich_lead)
    def score_lead(self):
        tier = ask("Grade this company as 'hot', 'warm', or 'cold' for enterprise B2B sales:\n"
                   f"{self.state.enrichment}\nReturn ONLY that single word.").lower()
        self.state.tier = "hot" if "hot" in tier else "warm" if "warm" in tier else "cold"
        return self.state.tier

    @listen("hot")
    def run_outreach_crew(self):
        return _one_task("Senior SDR", "Write bespoke executive pitch email", "Enterprise sales specialist.",
                         f"Write a high-touch cold email to leadership at {self.state.company} using:\n{self.state.enrichment}",
                         "Cold email draft.")

    @listen("warm")
    def add_to_nurture(self):
        return f"Lead {self.state.company} added to marketing nurture list."

    @listen("cold")
    def archive_lead(self):
        return f"Lead {self.state.company} archived in CRM due to low ICP fit."

    @listen(or_(run_outreach_crew, add_to_nurture, archive_lead))
    def log_lead(self, action_result):
        return f"Company: {self.state.company} | Tier: {self.state.tier.upper()} | Action: {action_result}"


def run_agent9(text: str) -> str:
    flow = LeadQualificationFlow()
    m = re.search(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|io|ai|co|org|net|in)\b\S*)", text)
    flow.state.website = m.group(1).strip(",;)") if m else "not provided"
    company = text.replace(m.group(1), "") if m else text
    flow.state.company = re.sub(r"[\s,;()\-:|]+$", "", company.strip()) or text
    return flow.kickoff()


@app.post("/agent9", response_model=Answer, tags=["CrewAI"])
def agent9(q: Query):
    return safe(run_agent9, q.input)


# =====================================================================
# 10. Approval Report Flow
#     input: "topic"  or  "topic ... Feedback: critique text"
# =====================================================================
class ReportState(BaseModel):
    topic: str = ""
    draft: str = ""
    decision: str = "approved"


class ApprovalReportFlow(Flow[ReportState]):
    @start()
    def collect_topic(self):
        return self.state.topic

    @listen(collect_topic)
    def research_and_write(self, topic):
        r = make_agent("Analyst", "Research the topic", "Tech analyst.")
        w = make_agent("Writer", "Draft a report on the topic", "Tech journalist.")
        t1 = Task(description=f"Provide 3 core insights on {topic}", expected_output="Bullet insights.", agent=r)
        t2 = Task(description=f"Draft a report on {topic} based on the research.",
                  expected_output="Report draft.", agent=w, context=[t1])
        self.state.draft = str(Crew(agents=[r, w], tasks=[t1, t2], process=Process.sequential, verbose=False).kickoff())
        return self.state.draft

    @router(research_and_write)
    def check_decision(self):
        return "approved" if self.state.decision.strip().lower() == "approved" else "revise"

    @listen("approved")
    def publish_report(self):
        return f"REPORT PUBLISHED (Approved):\n\n{self.state.draft}"

    @listen("revise")
    def run_revision_crew(self):
        revised = _one_task("Editor", "Revise report based on feedback", "Senior editor.",
                            f"Revise the report based on this feedback: '{self.state.decision}'.\n"
                            f"Original draft:\n{self.state.draft}", "Revised draft.")
        return f"REPORT REVISED (Critique: '{self.state.decision}'):\n\n{revised}"


def run_agent10(text: str) -> str:
    topic, feedback = split_on(text, "feedback")
    flow = ApprovalReportFlow()
    flow.state.topic = topic or text
    flow.state.decision = feedback or "approved"
    return flow.kickoff()


@app.post("/agent10", response_model=Answer, tags=["CrewAI"])
def agent10(q: Query):
    return safe(run_agent10, q.input)


def _usage(i: int):
    def handler():
        return {
            "route": f"/agent{i}",
            "message": "This endpoint works with POST, not GET.",
            "send": {"input": "your text here"},
            "returns": {"output": "..."},
        }
    return handler


for _i in range(1, 11):
    app.add_api_route(f"/agent{_i}", _usage(_i), methods=["GET"], include_in_schema=False)