import os
import warnings
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Read API Key from environment variable
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key", "")
os.environ["GEMINI_API_KEY"] = api_key

from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import Flow, start, listen, router, or_, and_
from crewai.tools import tool
from duckduckgo_search import DDGS

# Shared Gemini LLM Instance (Free tier, stable endpoint)
llm = LLM(
    model="gemini/gemini-3.1-flash-lite-preview",
    api_key=api_key
)

# Free DuckDuckGo search tool
@tool("DuckDuckGoSearch")
def duckduckgo_search(query: str) -> str:
    """Searches DuckDuckGo and returns concise factual snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No web results found."
            return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Search fallback: {str(e)}"

app = FastAPI(
    title="CrewAI Assignment Server",
    description="FastAPI service hosting 10 CrewAI projects",
    version="1.0.0"
)

# Strict Request/Response Schemas matching the assignment specifications
class QueryRequest(BaseModel):
    input: str

class QueryResponse(BaseModel):
    output: str

@app.get("/")
def root():
    return {"status": "online", "docs": "/docs"}


# =====================================================================
# AGENT 1: Blog Post Crew (Sequential Crew)
# =====================================================================
@app.post("/agent1", response_model=QueryResponse)
async def agent1_blog_crew(payload: QueryRequest):
    try:
        researcher = Agent(
            role="Topic Researcher",
            goal="Find key factual developments on {topic}",
            backstory="You gather accurate insights for technical writing.",
            tools=[duckduckgo_search],
            llm=llm,
            verbose=False
        )
        writer = Agent(
            role="Blog Writer",
            goal="Draft a clean 500-word blog post based on research",
            backstory="You write engaging, educational markdown posts.",
            llm=llm,
            verbose=False
        )
        editor = Agent(
            role="Editor",
            goal="Polish the draft for tone and clarity",
            backstory="You ensure accuracy and clear formatting.",
            llm=llm,
            verbose=False
        )
        t1 = Task(description="Research '{topic}' and list 4 key points.", expected_output="Bullet list of facts.", agent=researcher)
        t2 = Task(description="Write a 500-word blog post based on t1.", expected_output="Full markdown article.", agent=writer)
        t3 = Task(description="Polish and finalize the article.", expected_output="Final polished markdown article.", agent=editor)

        crew = Crew(agents=[researcher, writer, editor], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)
        result = await crew.kickoff_async(inputs={"topic": payload.input})
        return QueryResponse(output=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 2: Competitor Analysis Crew (Hierarchical Crew)
# =====================================================================
@app.post("/agent2", response_model=QueryResponse)
async def agent2_competitor_analysis(payload: QueryRequest):
    try:
        scraper = Agent(role="Feature Analyst", goal="Extract features for {competitors}", backstory="Analyzes product specs.", tools=[duckduckgo_search], llm=llm, verbose=False)
        pricing = Agent(role="Pricing Specialist", goal="Analyze pricing tiers for {competitors}", backstory="Monetization expert.", tools=[duckduckgo_search], llm=llm, verbose=False)
        news = Agent(role="News Scout", goal="Track recent updates on {competitors}", backstory="Tracks market moves.", tools=[duckduckgo_search], llm=llm, verbose=False)
        strategist = Agent(role="Chief Strategist", goal="Build a SWOT matrix comparing {competitors}", backstory="Synthesizes SWOT battlecards.", llm=llm, verbose=False)

        t1 = Task(description="Analyze features of {competitors}.", expected_output="Features list.", agent=scraper)
        t2 = Task(description="Analyze pricing of {competitors}.", expected_output="Pricing table.", agent=pricing)
        t3 = Task(description="Gather recent news on {competitors}.", expected_output="News summary.", agent=news)
        t4 = Task(description="Produce a markdown SWOT table comparing {competitors}.", expected_output="Complete SWOT analysis.", agent=strategist, context=[t1, t2, t3])

        crew = Crew(agents=[scraper, pricing, news, strategist], tasks=[t1, t2, t3, t4], process=Process.hierarchical, manager_llm=llm, verbose=False)
        result = await crew.kickoff_async(inputs={"competitors": payload.input})
        return QueryResponse(output=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 3: Meeting Notes Crew (Parallel Tasks)
# =====================================================================
@app.post("/agent3", response_model=QueryResponse)
async def agent3_meeting_notes(payload: QueryRequest):
    try:
        summarizer = Agent(role="Summarizer", goal="Create 5-bullet summary", backstory="Minutes recorder.", llm=llm, verbose=False)
        action_finder = Agent(role="Action Tracker", goal="Extract action items with owners and due dates", backstory="Task tracker.", llm=llm, verbose=False)
        decision_logger = Agent(role="Decision Logger", goal="Log agreed decisions", backstory="Consensus tracker.", llm=llm, verbose=False)
        formatter = Agent(role="Formatter", goal="Assemble final meeting document", backstory="Executive secretary.", llm=llm, verbose=False)

        t_sum = Task(description="Summarize: {transcript}", expected_output="5-bullet summary.", agent=summarizer, async_execution=True)
        t_act = Task(description="Extract Action | Owner | Due date from: {transcript}", expected_output="Markdown table of actions.", agent=action_finder, async_execution=True)
        t_dec = Task(description="Extract agreed decisions from: {transcript}", expected_output="Bullet list of decisions.", agent=decision_logger, async_execution=True)
        t_fmt = Task(
            description="Assemble summary, actions table, and decisions into clean markdown with headings.",
            expected_output="Formatted meeting notes.",
            agent=formatter,
            context=[t_sum, t_act, t_dec]
        )

        crew = Crew(agents=[summarizer, action_finder, decision_logger, formatter], tasks=[t_sum, t_act, t_dec, t_fmt], verbose=False)
        result = await crew.kickoff_async(inputs={"transcript": payload.input})
        return QueryResponse(output=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 4: Data Report Crew (Structured Output)
# =====================================================================
class SalesMetrics(BaseModel):
    total_revenue: float
    total_units_sold: int
    top_selling_product: str
    key_drivers: List[str]

@app.post("/agent4", response_model=QueryResponse)
async def agent4_data_report(payload: QueryRequest):
    try:
        analyst = Agent(role="Data Analyst", goal="Extract numerical metrics from text", backstory="Quantitative analyst.", llm=llm, verbose=False)
        writer = Agent(role="Report Writer", goal="Write strategic business insights", backstory="BI author.", llm=llm, verbose=False)
        reviewer = Agent(role="Reviewer", goal="Audit numbers and finalize report", backstory="Quality auditor.", llm=llm, verbose=False)

        t1 = Task(description="Extract metrics from data:\n{data}", expected_output="SalesMetrics schema.", agent=analyst, output_pydantic=SalesMetrics)
        t2 = Task(description="Explain top 3 commercial insights from metrics.", expected_output="3 strategic paragraphs.", agent=writer)
        t3 = Task(description="Audit numbers and assemble full markdown report.", expected_output="Complete audited report.", agent=reviewer, context=[t1, t2])

        crew = Crew(agents=[analyst, writer, reviewer], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)
        result = await crew.kickoff_async(inputs={"data": payload.input})
        return QueryResponse(output=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 5: Customer Support Flow (Router & or_)
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
        prompt = f"Categorize into one word: 'billing', 'access', or 'other'. Message: {body}"
        self.state.category = llm.call([{"role": "user", "content": prompt}]).strip().lower()
        return self.state.category

    @router(categorize_ticket)
    def route_by_category(self):
        if "billing" in self.state.category:
            return "billing"
        elif "access" in self.state.category:
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

    @listen(or_("billing", "access", "other"))
    def resolve_ticket(self, action):
        self.state.resolution = action
        return f"Category: {self.state.category.upper()} | Priority: {self.state.priority.upper()} | Action: {action}"

@app.post("/agent5", response_model=QueryResponse)
async def agent5_customer_support(payload: QueryRequest):
    try:
        flow = CustomerSupportFlow()
        flow.state.message = payload.input
        res = await flow.kickoff_async()
        return QueryResponse(output=str(res))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 6: Resume Screening Flow (Score Router)
# =====================================================================
class ResumeState(BaseModel):
    resume_text: str = ""
    score: int = 0
    decision: str = ""
    outcome: str = ""

class ResumeScreeningFlow(Flow[ResumeState]):
    @start()
    def load_resumes(self):
        return self.state.resume_text

    @listen(load_resumes)
    def score_resume(self, text):
        prompt = f"Evaluate this resume for a software engineering position. Return ONLY an integer score from 0 to 10.\nResume: {text}"
        raw = llm.call([{"role": "user", "content": prompt}]).strip()
        self.state.score = int(''.join(filter(str.isdigit, raw)) or 5)
        return self.state.score

    @router(score_resume)
    def check_score(self):
        return "shortlist" if self.state.score >= 7 else "reject"

    @listen("shortlist")
    def run_summary_crew(self):
        self.state.decision = "SHORTLISTED"
        evaluator = Agent(role="Recruiter", goal="List candidate strengths and gaps", backstory="Talent Lead.", llm=llm, verbose=False)
        task = Task(description=f"Summarize candidate strengths: {self.state.resume_text}", expected_output="Candidate dossier.", agent=evaluator)
        crew = Crew(agents=[evaluator], tasks=[task], verbose=False)
        self.state.outcome = str(crew.kickoff())
        return self.state.outcome

    @listen("reject")
    def draft_rejection(self):
        self.state.decision = "REJECTED"
        self.state.outcome = "Thank you for applying. We are moving forward with other applicants whose profiles align more closely with our needs."
        return self.state.outcome

    @listen(or_("shortlist", "reject"))
    def save_results(self, text):
        return f"Decision: {self.state.decision} ({self.state.score}/10)\n\n{text}"

@app.post("/agent6", response_model=QueryResponse)
async def agent6_resume_screening(payload: QueryRequest):
    try:
        flow = ResumeScreeningFlow()
        flow.state.resume_text = payload.input
        res = await flow.kickoff_async()
        return QueryResponse(output=str(res))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 7: Bug Triage Flow (3 Sub-Crews)
# =====================================================================
class IssueState(BaseModel):
    issue_text: str = ""
    issue_type: str = ""
    reply: str = ""

class BugTriageFlow(Flow[IssueState]):
    @start()
    def read_issue(self):
        return self.state.issue_text

    @listen(read_issue)
    def classify_issue(self, text):
        prompt = f"Classify this issue into 'bug', 'feature', or 'question'. Return ONLY that word.\n\n{text}"
        self.state.issue_type = llm.call([{"role": "user", "content": prompt}]).strip().lower()
        return self.state.issue_type

    @router(classify_issue)
    def route_type(self):
        if "bug" in self.state.issue_type:
            return "bug"
        elif "feature" in self.state.issue_type:
            return "feature"
        return "question"

    @listen("bug")
    def run_bug_crew(self):
        qa = Agent(role="QA Lead", goal="Identify severity and repro steps", backstory="Diagnostic engineer.", llm=llm, verbose=False)
        task = Task(description=f"Analyze bug: {self.state.issue_text}", expected_output="Severity (P1-P4) and repro steps.", agent=qa)
        return str(Crew(agents=[qa], tasks=[task], verbose=False).kickoff())

    @listen("feature")
    def run_feature_crew(self):
        pm = Agent(role="Product Manager", goal="Write User Story and Acceptance Criteria", backstory="Product lead.", llm=llm, verbose=False)
        task = Task(description=f"Draft user story for: {self.state.issue_text}", expected_output="Formatted user story.", agent=pm)
        return str(Crew(agents=[pm], tasks=[task], verbose=False).kickoff())

    @listen("question")
    def run_answer_crew(self):
        dev = Agent(role="Support Dev", goal="Provide step-by-step solution", backstory="Dev specialist.", llm=llm, verbose=False)
        task = Task(description=f"Answer: {self.state.issue_text}", expected_output="Step-by-step technical answer.", agent=dev)
        return str(Crew(agents=[dev], tasks=[task], verbose=False).kickoff())

    @listen(or_("bug", "feature", "question"))
    def post_reply(self, reply):
        self.state.reply = reply
        return f"TRIAGE [{self.state.issue_type.upper()}]:\n\n{reply}"

@app.post("/agent7", response_model=QueryResponse)
async def agent7_bug_triage(payload: QueryRequest):
    try:
        flow = BugTriageFlow()
        flow.state.issue_text = payload.input
        res = await flow.kickoff_async()
        return QueryResponse(output=str(res))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 8: Content Repurposing Flow (Parallel Sub-Crews + and_)
# =====================================================================
class RepurposeState(BaseModel):
    article: str = ""
    points: str = ""
    linkedin: str = ""
    twitter: str = ""
    newsletter: str = ""

class ContentRepurposeFlow(Flow[RepurposeState]):
    @start()
    def load_article(self):
        return self.state.article

    @listen(load_article)
    def extract_points(self, article):
        prompt = f"Extract 3 core takeaways from this text:\n{article}"
        self.state.points = llm.call([{"role": "user", "content": prompt}]).strip()
        return self.state.points

    @listen(extract_points)
    def make_linkedin(self, points):
        agent = Agent(role="LinkedIn Writer", goal="Write 150-word post", backstory="B2B copywriter.", llm=llm, verbose=False)
        task = Task(description=f"Draft LinkedIn post based on:\n{points}", expected_output="LinkedIn post with hook and hashtags.", agent=agent)
        self.state.linkedin = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        return self.state.linkedin

    @listen(extract_points)
    def make_twitter(self, points):
        agent = Agent(role="Twitter Creator", goal="Write a 3-tweet thread", backstory="Tech writer.", llm=llm, verbose=False)
        task = Task(description=f"Write a 3-tweet thread (1/3, 2/3, 3/3) based on:\n{points}", expected_output="Numbered tweet thread.", agent=agent)
        self.state.twitter = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        return self.state.twitter

    @listen(extract_points)
    def make_newsletter(self, points):
        agent = Agent(role="Newsletter Editor", goal="Write a quick newsletter blurb", backstory="Email curator.", llm=llm, verbose=False)
        task = Task(description=f"Write newsletter blurb with intro and 3 takeaways based on:\n{points}", expected_output="Newsletter section.", agent=agent)
        self.state.newsletter = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        return self.state.newsletter

    @listen(and_(make_linkedin, make_twitter, make_newsletter))
    def publish_pack(self, outputs):
        return (
            f"## LinkedIn Post\n{self.state.linkedin}\n\n"
            f"## Twitter Thread\n{self.state.twitter}\n\n"
            f"## Newsletter Section\n{self.state.newsletter}"
        )

@app.post("/agent8", response_model=QueryResponse)
async def agent8_content_repurposing(payload: QueryRequest):
    try:
        flow = ContentRepurposeFlow()
        flow.state.article = payload.input
        res = await flow.kickoff_async()
        return QueryResponse(output=str(res))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 9: Lead Qualification Flow
# =====================================================================
class LeadState(BaseModel):
    lead_info: str = ""
    enrichment: str = ""
    tier: str = ""

class LeadQualificationFlow(Flow[LeadState]):
    @start()
    def capture_lead(self):
        return self.state.lead_info

    @listen(capture_lead)
    def enrich_lead(self, info):
        agent = Agent(role="B2B Researcher", goal="Find business domain and scale", backstory="Lead researcher.", llm=llm, verbose=False)
        task = Task(description=f"Summarize business model and scale for: {info}", expected_output="3-point company brief.", agent=agent)
        self.state.enrichment = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        return self.state.enrichment

    @router(enrich_lead)
    def score_lead(self):
        prompt = f"Grade this company as 'hot', 'warm', or 'cold' for enterprise B2B sales:\n{self.state.enrichment}\nReturn ONLY that single word."
        tier = llm.call([{"role": "user", "content": prompt}]).strip().lower()
        if "hot" in tier:
            self.state.tier = "hot"
            return "hot"
        elif "warm" in tier:
            self.state.tier = "warm"
            return "warm"
        self.state.tier = "cold"
        return "cold"

    @listen("hot")
    def run_outreach_crew(self):
        sdr = Agent(role="Senior SDR", goal="Write bespoke executive pitch email", backstory="Enterprise sales specialist.", llm=llm, verbose=False)
        task = Task(description=f"Write high-touch cold email to leadership using:\n{self.state.enrichment}", expected_output="Cold email draft.", agent=sdr)
        return str(Crew(agents=[sdr], tasks=[task], verbose=False).kickoff())

    @listen("warm")
    def add_to_nurture(self):
        return "Lead added to middle-funnel marketing nurture campaign."

    @listen("cold")
    def archive_lead(self):
        return "Lead cataloged as low fit and archived in CRM."

    @listen(or_("hot", "warm", "cold"))
    def log_lead(self, action_result):
        return f"Lead: {self.state.lead_info} | Tier: {self.state.tier.upper()} | Result: {action_result}"

@app.post("/agent9", response_model=QueryResponse)
async def agent9_lead_qualification(payload: QueryRequest):
    try:
        flow = LeadQualificationFlow()
        flow.state.lead_info = payload.input
        res = await flow.kickoff_async()
        return QueryResponse(output=str(res))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# AGENT 10: Approval Report Flow
# =====================================================================
class ReportState(BaseModel):
    topic: str = ""
    draft: str = ""

class ApprovalReportFlow(Flow[ReportState]):
    @start()
    def collect_topic(self):
        return self.state.topic

    @listen(collect_topic)
    def research_and_write(self, topic):
        researcher = Agent(role="Analyst", goal="Research {topic}", backstory="Tech analyst.", llm=llm, verbose=False)
        writer = Agent(role="Writer", goal="Draft structured report", backstory="Tech author.", llm=llm, verbose=False)
        t1 = Task(description=f"Provide 3 core insights on {topic}", expected_output="Bullet insights.", agent=researcher)
        t2 = Task(description=f"Draft report on {topic} based on research.", expected_output="Report draft.", agent=writer)
        self.state.draft = str(Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.sequential, verbose=False).kickoff())
        return self.state.draft

    @listen(research_and_write)
    def finalize_report(self, draft):
        return f"=== APPROVED REPORT ===\n\nTopic: {self.state.topic}\n\n{draft}"

class Route10Input(BaseModel):
    input: str  # Format: "Topic: Quantum Computing | Review: approved" OR "Topic: Quantum Computing | Review: Add a section on cost"

@app.post("/agent10", response_model=QueryResponse)
async def agent10_approval_report(payload: QueryRequest):
    try:
        # Check if the user supplied feedback inside the input string
        user_text = payload.input
        if "|" in user_text:
            topic, feedback = [part.strip() for part in user_text.split("|", 1)]
        else:
            topic = user_text
            feedback = "approved"  # Default assumption if only topic is given

        flow = ApprovalReportFlow()
        flow.state.topic = topic
        flow.state.decision = feedback
        res = await flow.kickoff_async()
        return QueryResponse(output=str(res))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))