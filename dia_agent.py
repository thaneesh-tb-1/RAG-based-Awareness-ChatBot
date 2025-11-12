"""
DIA - Dementia Information Assistant
A CLI chatbot powered by Google Gemini with RAG retrieval from Vertex AI
"""

import os
import sys
import re
import csv
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types
from urllib.parse import quote

# Try to import requests, fallback if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[Warning: requests library not found. Web search for citations will be limited.]")

# Load environment variables
load_dotenv()

# CSV Logging configuration
LOG_FILE = "dia_query_log.csv"
MAX_LOG_LINES = 500

# Category priority for fallback (higher priority first)
CATEGORY_PRIORITY = [
    "policy_support",
    "community_awareness",
    "caregiving",
    "management",
    "lifestyle_prevention",
    "research_innovation",
    "stigma_awareness",
    "resources_helplines",
    "care_guides",
    "core_medical"
]

# System prompt for DIA
SYSTEM_PROMPT = """You are a helpful, knowledgeable companion who understands dementia and can talk about it naturally and supportively. You're here to have genuine conversations, not to sound like a textbook or a formal assistant.

Your goal is to help people understand dementia through natural, warm conversations. Use only verified information from trusted health sources (WHO, Alzheimer's Association, NIMHANS, Dementia India Alliance, NHS). Never invent facts.

CONVERSATION STYLE - BE NATURAL AND HUMAN:
- Match the user's vibe: If they're casual ("got it"), be casual back. If they're formal, match that. If they're worried, be gentle and reassuring. Mirror their energy and tone naturally.
- Sound like a real person: Write as if you're texting a friend who asked you about dementia. Use natural flow, not rigid structures.
- Avoid robotic phrases: Don't say things like "I am an AI assistant" or "My purpose is..." or "I'm designed to..." Just be helpful naturally.
- Be conversational: Use contractions (it's, that's, you're), natural transitions, and varied sentence lengths. Sometimes short sentences. Sometimes longer ones. Like real speech.
- Don't over-explain: Trust that the user understands. If they say "got it," just acknowledge briefly - don't launch into another explanation unless they ask.

FORMATTING - MAKE IT EASY TO READ AND NICE:
- Break up long paragraphs: Don't write walls of text. Use short paragraphs (2-3 sentences max) to make it less overwhelming.
- Use headings when explaining multiple concepts: Use **bold headings** or emoji + text to break up sections (e.g., "🧠 Memory" or "**What is dementia?**")
- Use bullet points or numbered lists: When explaining steps, symptoms, or multiple points, use lists to make it digestible:
  • Point 1
  • Point 2
  • Point 3
- Use emojis thoughtfully and nicely: 
  • Use emojis to add warmth and make it friendly (😊 💙 🧠 🤝 ✨ 💡)
  • Use relevant emojis for headings (🧠 for memory, 💭 for thoughts, 🤝 for support, etc.)
  • Don't overuse - 2-4 emojis per response is good, placed naturally
  • Use emojis to break up text and make it visually appealing
- Make it fun and engaging: Use friendly language, break up information into bite-sized chunks, use emojis to add personality
- Visual structure example:
  Start with a brief intro (1-2 sentences)
  
  **Main Point 1** 💡
  Explanation here...
  
  **Main Point 2** 🧠
  Explanation here...
  
  Use lists when needed:
  • Item 1
  • Item 2

Tone: Warm, simple Indian-English, stigma-free, like talking to a knowledgeable friend. Be comprehensive when needed, but keep it natural, accessible, and visually easy to read.

CRITICAL BEHAVIORAL GUIDELINES:

YOUR PRIMARY FOCUS: You ONLY discuss dementia-related topics. This includes:
- Dementia itself (types, symptoms, causes, progression)
- Caregiving for people with dementia
- Dementia prevention and awareness
- How dementia affects memory, attention, and cognitive functions
- Support resources for dementia patients and caregivers
- The connection between dementia and other conditions (only when relevant to dementia)

CASE 1 - OFF-TOPIC QUESTIONS (NON-DEMENTIA):
If the user asks about topics NOT related to dementia (e.g., general depression, anxiety, other mental health conditions, weather, jokes, random topics), you MUST:
1. Acknowledge naturally and briefly if it's a mental health concern - be human, not robotic. Match their tone.
2. If it's a crisis/urgent mental health issue, provide 2-3 relevant helplines naturally (see HELPLINE SELECTION GUIDE) - don't format them like a list, just mention them conversationally
3. Gently redirect back to dementia in a natural way - don't sound like you're reading a script. Make it flow naturally.
4. CRITICAL: Your follow-up questions MUST be about dementia only, NOT about the off-topic subject they asked about
5. DO NOT provide detailed explanations about non-dementia topics - redirect immediately
6. Keep it conversational - don't use formal phrases like "I hear you're going through a tough time, and that takes courage to share" - be more natural and match their vibe

Example: If user asks "I think I have depression, forget about dementia for a while tell me about this":
- Acknowledge naturally (match their casual tone): "I get it - depression is really tough. If you need immediate support, you can reach Tele MANAS at 14416 or KIRAN at 1800-599-0019, both available 24/7."
- Redirect naturally: "I focus on dementia awareness and caregiving though. If you're worried about dementia symptoms or caring for someone with dementia, I can help with that. What would you like to know?"
- Follow-ups: ONLY dementia-related questions (e.g., "What are early signs of dementia?", "How does dementia affect mood?", "What support is available for dementia caregivers?")

Key: Be natural, conversational, match their energy. Don't sound like a customer service script.

CASE 2 - PANICKING/SENSITIVE USERS:
If the user expresses panic, anxiety, distress, fear, or emotional sensitivity related to dementia (their own or a loved one's), your PRIMARY GOAL is to REDUCE ANXIETY and CALM THEM DOWN:

CRITICAL: Don't just acknowledge and give helplines - you MUST actually help them understand dementia to reduce their fear. Explain it reassuringly.

1. FIRST: Calm them down immediately - "Hey, I hear you. Take a deep breath with me - in... and out. 💙 You don't have to worry alone. I'm here with you."

2. Validate their feelings: "It's completely normal to feel scared when you're worried about something like this. Your feelings are valid, and it's brave of you to share this."

3. THEN: Actually explain dementia in a reassuring, calming way. This is CRITICAL - don't skip this:
   - Explain what dementia is simply and reassuringly
   - Mention that many things can cause memory issues (stress, lack of sleep, medications, etc.) - it's not always dementia
   - Explain that dementia is progressive and usually develops slowly over time
   - Reassure that getting checked by a doctor is the right step
   - Use headings, bullet points, and emojis to make it easy to read and less overwhelming
   - Format it nicely so it's not a wall of text

4. THEN provide RELEVANT helpline numbers naturally (see HELPLINE SELECTION GUIDE below) - mention them conversationally, not as a formal list

5. Be encouraging, supportive, and use a calm, empathetic tone throughout

6. Format it nicely: Use headings, bullet points, and emojis to break up the information and make it less overwhelming

Example structure for "I think I have dementia I am scared":
- Start: "Hey, I hear you. Take a breath with me - in... and out. 💙 You don't have to worry alone."
- Validate: "It's completely normal to feel scared. Your feelings are valid."
- Explain dementia reassuringly with headings and formatting:
  **What is dementia?** 🧠
  Brief, reassuring explanation...
  
  **Important things to know** 💡
  • Many things can cause memory issues...
  • Dementia usually develops slowly...
  • Getting checked is the right step...
  
- Then helplines: "If you need someone to talk to, you can reach..."
- Be supportive throughout, not robotic

CASE 3 - IMMEDIATE CRISIS/PANIC:
If the user expresses immediate panic, fear, crisis, suicidal thoughts, or severe distress, your PRIMARY GOAL is to IMMEDIATELY CALM THEM and provide CRITICAL helplines:
1. FIRST: Immediate calming - "I'm here with you right now. Let's breathe together - in... and out. You're safe. 💙"
2. Provide ONLY the most critical helplines (2-3 numbers max) based on their specific crisis:
   - If mental health crisis/suicidal: Provide Tele MANAS (14416) and KIRAN (1800-599-0019)
   - If medical emergency: Provide Universal Emergency (112) and Ambulance (108)
   - If dementia-specific crisis: Provide NIMHANS (080-46110007) and ARDSI (9846198471)
3. Format simply and clearly - don't overwhelm with too many numbers
4. Reassure: "These services are available right now, 24/7, and completely confidential. Please reach out - you don't have to go through this alone."
5. DO NOT provide all helplines at once - only the most relevant 2-3 for their specific situation

RESPONSE STYLE - BE NATURALLY HELPFUL:
- Match their energy: If they're brief ("got it"), be brief back. If they're asking deep questions, give thoughtful answers. Mirror their communication style.
- Be encouraging and validating: Acknowledge feelings naturally, normalize experiences without being preachy
- Use easy language: Explain things simply, like you would to a friend. Avoid medical jargon unless necessary, then explain it.
- Be conversational: Write like you're having a real conversation, not delivering a presentation. Use natural flow, varied sentence structure.
- Don't sound like a bot: Avoid phrases like "I understand you're asking about..." or "Let me explain..." or "I hear you're going through..." Just answer naturally, as if you're a knowledgeable friend.
- When someone is scared or panicking: DON'T just acknowledge and give helplines then leave. Actually explain dementia reassuringly to help reduce their anxiety. Help them understand what dementia is, what causes memory issues, and reassure them. Then provide helplines. Be supportive throughout - don't give up on them.
- When redirecting off-topic: Make it flow naturally. Don't use formal acknowledgment phrases. Just mention helplines conversationally if needed, then smoothly redirect.
- Format for readability: Break up information with headings, bullet points, and emojis. Make it visually appealing and easy to scan. Don't write long paragraphs - break them up.
- Use emojis nicely: Use 2-4 emojis per response when they add value - in headings, to emphasize points, or to add warmth. Make it fun and engaging.
- Read the room: If they seem satisfied ("got it"), don't over-explain. If they're confused or scared, be more detailed and reassuring, but still format it nicely.

HELPLINE INFORMATION (USE EXACT NUMBERS - DO NOT HALLUCINATE):

CRITICAL: When providing helplines, ALWAYS provide ONLY 2-3 most relevant numbers based on the user's specific situation. DO NOT dump all numbers at once.

HELPLINE SELECTION GUIDE:
- Mental Health Crisis/Suicidal Thoughts: Tele MANAS (14416) + KIRAN (1800-599-0019)
- Medical Emergency: Universal Emergency (112) + Ambulance (108)
- Dementia-Specific Crisis: NIMHANS (080-46110007) + ARDSI (9846198471)
- General Anxiety/Panic: Tele MANAS (14416) + Dementia India Alliance (8585 990 990)
- Senior Citizen Support: Elder Line (14567) + HelpAge India (1800-180-1253)
- Caregiver Support: ARDSI (9846198471) + Dementia India Alliance (8585 990 990)

National Dementia & Mental Health Helplines:
• NIMHANS (Bengaluru, 24/7): 080-46110007 | Main Office: 080-26995000
• ARDSI National: 9846198471, 9846198473, 9846198786 | Landline: +91 4885 223801
• Dementia India Alliance: 8585 990 990 (8 AM-6 PM, Mon-Sat)
• Tele MANAS (24/7, 365 days): 14416 or 1800-89-14416

Emergency & Medical Services:
• Universal Emergency: 112 (24/7 - Police, Fire, Medical)
• Ambulance: 108 (24/7) or 102 (Maternal/Child Health, 24/7)
• Fire Brigade: 101 (24/7)
• Police Emergency: 100 (24/7)

Mental Health Crisis & Suicide Prevention:
• KIRAN (24/7): 1800-599-0019
• Vandrevala Foundation (24/7): 9999 666 555
• iCALL (TISS Mumbai, Mon-Sat 8 AM-10 PM): 9152987821 or 022-25521111
• AASRA (Mumbai, 24/7): 022-27546669
• Sneha (Chennai, 24/7): 044-24640050, 044-24640060
• Mpower Minds (24/7): 1800-120-820050

Senior Citizen Support:
• Elder Line (8 AM-8 PM, 7 days): 14567 (Toll-free)
• HelpAge India: 1800-180-1253

Palliative Care:
• Saath Saath (Mon-Sat 10 AM-6 PM): 1800-202-7777
• Pallium India (Mon-Sat 9:30 AM-4 PM): +91 860 688 4889 or +91 964 588 4889

IMPORTANT CONVERSATION RULES:
- After every response (except in CASE 3 immediate crisis where helplines are provided, or when user gives brief acknowledgments like "got it", "thanks", "okay"), you MUST provide 2-3 relevant follow-up questions. Format these questions at the end of your response like this:

---
**You might also want to know:**
1. [Question 1]
2. [Question 2]
3. [Question 3]

- CRITICAL: Follow-up questions MUST ALWAYS be about dementia, caregiving, or dementia-related topics. NEVER provide follow-up questions about off-topic subjects (depression, anxiety, etc.) - always redirect to dementia.
- If the conversation has already started (you've already sent a welcome message), DO NOT greet the user again. Just answer their question directly and naturally. Only greet if this is the very first message in a completely new conversation.
- Match their vibe: If they're casual, be casual. If they're formal, be formal. If they're worried, be gentle. Mirror their communication style naturally.
- Don't expose your purpose: Never say things like "I'm an AI assistant" or "My purpose is..." or "I'm designed to help with..." Just be helpful naturally, like a knowledgeable friend.
- Keep it natural: If they say "got it" or "thanks" or similar brief acknowledgments, acknowledge briefly and offer follow-up questions, but don't launch into another long explanation unless they ask.
- STAY ON TOPIC: You ONLY discuss dementia-related topics. If users ask about other mental health conditions, acknowledge briefly, provide helplines if urgent, but redirect back to dementia. Do not provide detailed explanations about non-dementia topics.
- FORMAT NICELY: Always use headings, bullet points, and emojis to make responses easy to read and less overwhelming. Break up paragraphs, use visual structure. Make it fun and engaging to read.
- SOURCES: When you use information from RAG retrieval, the system will automatically add source references in a "References" section before the "You might also want to know" section. You don't need to mention sources in your text - they'll be added automatically in standard citation format [1], [2], etc.
- Make sure the follow-up questions are relevant, helpful, and encourage further learning about DEMENTIA ONLY."""


class DIAAgent:
    """Dementia Information Assistant Agent"""

    def __init__(self):
        """Initialize the DIA agent with Google GenAI and RAG configuration"""
        # Get environment variables
        self.rag_corpus = os.getenv("RAG_CORPUS")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION")

        # Google Custom Search API credentials (for finding PDF links)
        self.google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not self.rag_corpus:
            raise ValueError("RAG_CORPUS not found in environment variables")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment variables")
        if not self.location:
            raise ValueError("GOOGLE_CLOUD_LOCATION not found in environment variables")

        # Initialize Google GenAI client with Vertex AI
        # Note: Vertex AI requires OAuth2 authentication (Application Default Credentials)
        # NOT API keys. Set up authentication using one of:
        # 1. gcloud auth application-default login
        # 2. GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to service account key
        self.client = genai.Client(
            vertexai=True,
        )

        # Model configuration
        # Using Gemini 2.5 Flash model
        # If gemini-2.5-flash doesn't work, try gemini-2.5-flash-preview-09-2025
        # Note: Model availability varies by region - may need to use us-central1 or global
        self.model = "gemini-2.5-flash"

        # RAG tool configuration with metadata-based filtering
        # Default category is None (retrieves from all categories)
        # Will be dynamically updated based on query intent
        self.current_category_filter = None
        self.top_k = 5  # Number of documents to retrieve

        # Base RAG tools - will be rebuilt with filters dynamically
        self.tools = self._build_rag_tools()

        # Generation configuration (optimized with 8192 token limit)
        self.generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
            tools=self.tools,
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
            system_instruction=SYSTEM_PROMPT,
        )

        # Generation configuration without RAG tools (for cached context responses)
        self.generate_content_config_no_rag = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
            system_instruction=SYSTEM_PROMPT,
        )

        # Conversation history
        self.conversation_history = []

        # Context cache: stores (query, response, keywords) for reuse
        self.context_cache: List[Dict[str, str]] = []

    def _build_rag_tools(self, category: str = None):
        """
        Build RAG tools configuration

        NOTE: The current Google GenAI SDK version doesn't support rag_retrieval_config
        with top_k and filter parameters directly in VertexRagStoreRagResource.
        This method is prepared for future SDK updates.

        For now, we'll use query enhancement to hint at categories.

        Args:
            category: "core", "community", "contextual", or None (not used yet)
        """
        # Simple working structure (no rag_retrieval_config for now)
        return [
            types.Tool(
                retrieval=types.Retrieval(
                    vertex_rag_store=types.VertexRagStore(
                        rag_resources=[
                            types.VertexRagStoreRagResource(
                                rag_corpus=self.rag_corpus
                            )
                        ],
                    )
                )
            )
        ]

    def _detect_categories_with_scores(self, query: str) -> List[Tuple[str, int]]:
        """
        Detect categories with weighted scoring
        Returns: List of (category, score) tuples sorted by score
        """
        query_lower = query.lower()

        # Enhanced categories with specific keywords
        CATEGORIES = {
            "core_medical": ["symptom", "type", "diagnosis", "alzheimer", "vascular", "memory",
                            "cognitive", "lewy", "frontotemporal", "stage", "stages",
                            "progression", "brain", "neurological", "medical", "doctor",
                            "disease", "dementia type", "what is dementia", "causes", "sign",
                            "signs", "confused", "confusion", "forgetful", "forget"],
            "management": ["treatment", "therapy", "routine", "sleep", "music", "activity", "rehab",
                          "medication", "medicine", "daily care", "activities", "schedule",
                          "behavioral", "wandering", "aggression", "sundowning", "managing"],
            "caregiving": ["caregiver", "family", "support", "stress", "safety", "respite",
                          "caregiving", "care", "helping", "coping", "burden",
                          "day care", "facility", "home care", "bathing", "feeding"],
            "policy_support": ["government", "rights", "scheme", "pension", "disability",
                              "policy", "benefit", "welfare", "certificate", "legal",
                              "guardian", "power of attorney", "social security"],
            "community_awareness": ["asha", "camp", "panchayat", "volunteer", "public", "school",
                                   "community", "awareness camp", "screening", "outreach",
                                   "rural", "urban", "neighborhood", "local", "village", "poster"],
            "lifestyle_prevention": ["diet", "yoga", "exercise", "sleep", "stress", "habit",
                                    "prevent", "prevention", "reduce risk", "lifestyle",
                                    "nutrition", "healthy", "activity", "social", "engagement",
                                    "brain health", "cognitive reserve", "learning", "risk factor",
                                    "avoid", "protect", "smoking", "alcohol"],
            "research_innovation": ["nimhans", "aiims", "trial", "app", "ai", "study",
                                   "research", "clinical trial", "technology", "innovation",
                                   "digital", "wearable", "monitoring", "detection"],
            "stigma_awareness": ["stigma", "myth", "story", "awareness", "campaign",
                                "discrimination", "misconception", "belief", "taboo",
                                "education", "advocacy", "media"],
            "resources_helplines": ["helpline", "phone", "contact", "support center",
                                   "hotline", "call", "emergency", "crisis",
                                   "assistance", "elderline", "ardsi", "helpline number"],
            "care_guides": ["manual", "guide", "checklist", "plan", "toolkit",
                           "handbook", "documentation", "care plan", "guideline", "instruction"]
        }

        # Count matches for each category
        category_scores = {}
        for category, keywords in CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            category_scores[category] = score

        # Sort by score (descending)
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_categories

    def _get_primary_category_with_fallback(self, categories_with_scores: List[Tuple[str, int]]) -> str:
        """
        Get primary category using priority fallback for ties

        Args:
            categories_with_scores: List of (category, score) tuples

        Returns:
            Primary category name or None
        """
        # Filter categories with scores > 0
        relevant_categories = [(cat, score) for cat, score in categories_with_scores if score > 0]

        if not relevant_categories:
            return None

        max_score = relevant_categories[0][1]

        # Get all categories with the max score
        top_categories = [cat for cat, score in relevant_categories if score == max_score]

        # If only one category, return it
        if len(top_categories) == 1:
            return top_categories[0]

        # If multiple categories have the same score, use priority fallback
        for priority_cat in CATEGORY_PRIORITY:
            if priority_cat in top_categories:
                print(f"[Debug: Using priority fallback: {priority_cat}]")
                return priority_cat

        # Fallback to first category (shouldn't reach here if CATEGORY_PRIORITY is complete)
        return top_categories[0]

    def _llm_classify_category(self, query: str) -> str:
        """
        Use LLM to classify query when keyword matching fails

        Args:
            query: User query

        Returns:
            Detected category or None
        """
        classification_prompt = f"""Which category does this query fall into:
"{query}"?

Choose from: core_medical, management, caregiving, lifestyle_prevention, community_awareness, policy_support, research_innovation, stigma_awareness, resources_helplines, care_guides.

Respond with ONLY the category name, nothing else."""

        try:
            print(f"[Debug: Using LLM for category classification]")
            response = self.client.models.generate_content(
                model=self.model,
                contents=[classification_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=50,
                )
            )

            detected_category = response.text.strip().lower()

            # Validate that it's a real category
            valid_categories = [
                "core_medical", "management", "caregiving", "lifestyle_prevention",
                "community_awareness", "policy_support", "research_innovation",
                "stigma_awareness", "resources_helplines", "care_guides"
            ]

            if detected_category in valid_categories:
                print(f"[Debug: LLM classified as: {detected_category}]")
                return detected_category
            else:
                print(f"[Debug: LLM returned invalid category: {detected_category}]")
                return None

        except Exception as e:
            print(f"[Debug: LLM classification failed: {str(e)}]")
            return None

    def _detect_query_category(self, query: str) -> str:
        """
        Detect the category of the user query with enhanced scoring and fallback

        Returns: category name or None (no filter)
        """
        # Get categories with scores
        categories_with_scores = self._detect_categories_with_scores(query)

        # Log top 3 categories
        print(f"\n[Debug: Top categories: {categories_with_scores[:3]}]")

        # Get primary category with priority fallback
        primary_category = self._get_primary_category_with_fallback(categories_with_scores)

        if primary_category:
            print(f"[Debug: Primary category detected: {primary_category.upper()}]")
            return primary_category

        # If no category detected, try LLM classification
        print(f"[Debug: No keyword match, attempting LLM classification]")
        llm_category = self._llm_classify_category(query)

        if llm_category:
            return llm_category

        print(f"[Debug: No category detected, using all documents]")
        return None

    def _enhance_query_with_category(self, query: str, category: str = None) -> str:
        """
        Enhance the query with category-specific keywords to improve retrieval relevance

        This is a workaround for SDK versions that don't support metadata filtering.
        When the SDK supports rag_retrieval_config with filters, this can be replaced.

        Args:
            query: Original user query
            category: Category name or None

        Returns:
            Enhanced query with category hints
        """
        if category is None:
            return query

        # Add category-specific context hints to improve retrieval
        category_hints = {
            "core_medical": "medical symptoms diagnosis dementia types memory cognitive",
            "management": "treatment therapy routine care management medication daily activities",
            "caregiving": "caregiver family support safety home care coping respite",
            "policy_support": "government pension disability rights scheme welfare policy legal",
            "community_awareness": "community ASHA camp village panchayat awareness outreach screening",
            "lifestyle_prevention": "prevention diet yoga exercise lifestyle brain health nutrition habits",
            "research_innovation": "research NIMHANS AIIMS trial innovation technology app AI study",
            "stigma_awareness": "stigma myth awareness discrimination education advocacy campaign",
            "resources_helplines": "helpline phone support center contact assistance emergency hotline",
            "care_guides": "manual toolkit guide care plan template checklist handbook instruction"
        }

        hint = category_hints.get(category, "")
        if hint:
            # Prepend the hint to help RAG focus on relevant documents
            enhanced = f"[Focus: {hint}] {query}"
            print(f"[Debug: Enhanced query for '{category}' category]")
            return enhanced

        return query

    def _log_query(self, user_query: str, detected_categories: List[Tuple[str, int]],
                   num_sources: int, used_cache: bool):
        """
        Log query details to CSV file

        Args:
            user_query: The user's query
            detected_categories: List of (category, score) tuples
            num_sources: Number of sources retrieved
            used_cache: Whether cache was used
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Format categories as comma-separated list of top 3
            top_categories = ",".join([f"{cat}({score})" for cat, score in detected_categories[:3]])

            # Create log entry
            log_entry = [timestamp, user_query, top_categories, num_sources, used_cache]

            # Check if file exists and get line count
            file_exists = os.path.exists(LOG_FILE)
            line_count = 0

            if file_exists:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f)

            # If approaching max lines, truncate old entries
            if line_count >= MAX_LOG_LINES:
                # Read existing entries
                with open(LOG_FILE, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    rows = list(reader)

                # Keep only the most recent entries (MAX_LOG_LINES - 100)
                rows = rows[-(MAX_LOG_LINES - 100):]

                # Rewrite file with truncated data
                with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    if header:
                        writer.writerow(header)
                    writer.writerows(rows)

            # Append new entry
            with open(LOG_FILE, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)

                # Write header if new file
                if not file_exists or line_count == 0:
                    writer.writerow(['timestamp', 'user_query', 'detected_categories', 'num_sources', 'used_cache'])

                writer.writerow(log_entry)

        except Exception as e:
            print(f"[Debug: Error logging query: {str(e)}]")

    def _extract_keywords(self, text: str) -> set:
        """Extract important keywords from text (simple implementation)"""
        # Common stopwords to ignore
        stopwords = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'into', 'through', 'during',
            'can', 'could', 'should', 'would', 'may', 'might', 'must', 'will',
            'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'our', 'their', 'this', 'that',
            'these', 'those', 'some', 'any', 'all', 'each', 'every', 'both'
        }

        # Simple keyword extraction: lowercase, split, remove stopwords and short words
        words = text.lower().replace('?', '').replace('!', '').replace(',', '').split()
        keywords = {word for word in words if word not in stopwords and len(word) > 3}
        return keywords

    def _extract_sources_from_response(self, response) -> List[str]:
        """
        Extract source citations from Vertex AI RAG response
        Checks multiple places: tool calls, grounding metadata, retrieval results
        
        Args:
            response: Full response object from generate_content
            
        Returns:
            List of source URLs/document references
        """
        sources = []
        seen_sources = set()
        
        try:
            if not response.candidates or len(response.candidates) == 0:
                return sources
                
            candidate = response.candidates[0]
            
            # Method 1: Check tool calls (Vertex RAG returns results in tool calls)
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Check for function calls or tool responses
                        if hasattr(part, 'function_call'):
                            # Extract from function call results
                            pass
                        if hasattr(part, 'function_response'):
                            func_response = part.function_response
                            # Check retrieval results in function response
                            if hasattr(func_response, 'retrieval_results'):
                                sources.extend(self._extract_from_retrieval_results(func_response.retrieval_results, seen_sources))
            
            # Method 2: Check grounding metadata
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                grounding = candidate.grounding_metadata
                sources.extend(self._extract_sources_from_grounding(grounding, seen_sources))
            
            # Method 3: Check for retrieval metadata directly in candidate
            if hasattr(candidate, 'retrieval_metadata'):
                retrieval_meta = candidate.retrieval_metadata
                sources.extend(self._extract_from_retrieval_metadata(retrieval_meta, seen_sources))
            
            # Method 4: Check tool_use parts (newer API structure)
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'tool_use'):
                            tool_use = part.tool_use
                            if hasattr(tool_use, 'retrieval'):
                                # This is a retrieval tool call
                                pass
            
            # Debug output
            print(f"\n[Debug: Extracted {len(sources)} sources from response]")
            if sources:
                print(f"[Debug: Sources: {sources[:3]}...]")  # Show first 3
            else:
                print(f"[Debug: Candidate attributes: {[attr for attr in dir(candidate) if not attr.startswith('_')][:15]}]")
                if hasattr(candidate, 'content'):
                    print(f"[Debug: Content type: {type(candidate.content)}]")
                    if hasattr(candidate.content, 'parts'):
                        print(f"[Debug: Number of parts: {len(candidate.content.parts)}]")
                        for i, part in enumerate(candidate.content.parts[:2]):  # Check first 2 parts
                            print(f"[Debug: Part {i} type: {type(part)}, attributes: {[attr for attr in dir(part) if not attr.startswith('_')][:10]}]")
                                
        except Exception as e:
            print(f"[Debug: Error extracting sources from response: {str(e)}]")
            import traceback
            traceback.print_exc()
        
        return sources
    
    def _extract_from_retrieval_results(self, retrieval_results, seen_sources: set) -> List[str]:
        """Extract sources from retrieval_results object"""
        sources = []
        try:
            if hasattr(retrieval_results, 'contexts'):
                for ctx in retrieval_results.contexts:
                    # Check for document URI or source
                    for attr in ['uri', 'source_uri', 'document_uri', 'source', 'file_uri']:
                        if hasattr(ctx, attr):
                            source = getattr(ctx, attr)
                            if source and isinstance(source, str) and source not in seen_sources:
                                seen_sources.add(source)
                                sources.append(source)
        except Exception as e:
            print(f"[Debug: Error in _extract_from_retrieval_results: {str(e)}]")
        return sources
    
    def _extract_from_retrieval_metadata(self, retrieval_meta, seen_sources: set) -> List[str]:
        """Extract sources from retrieval_metadata object"""
        sources = []
        try:
            # Check various possible attributes
            for attr in ['sources', 'documents', 'contexts', 'retrieved_documents']:
                if hasattr(retrieval_meta, attr):
                    items = getattr(retrieval_meta, attr)
                    if items:
                        for item in items:
                            for source_attr in ['uri', 'source_uri', 'document_uri', 'source', 'file_uri', 'url']:
                                if hasattr(item, source_attr):
                                    source = getattr(item, source_attr)
                                    if source and isinstance(source, str) and source not in seen_sources:
                                        seen_sources.add(source)
                                        sources.append(source)
        except Exception as e:
            print(f"[Debug: Error in _extract_from_retrieval_metadata: {str(e)}]")
        return sources
    
    def _extract_sources_from_grounding(self, grounding, seen_sources: set = None) -> List[Dict]:
        """
        Extract source URLs and metadata from grounding metadata

        Args:
            grounding: Grounding metadata object from response
            seen_sources: Set of already seen sources to avoid duplicates

        Returns:
            List of source dictionaries with URI and metadata
        """
        sources = []
        if seen_sources is None:
            seen_sources = set()

        try:
            # Check for grounding_chunks
            if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                for chunk in grounding.grounding_chunks:
                    # Try retrieved_context (Vertex RAG)
                    if hasattr(chunk, 'retrieved_context'):
                        context = chunk.retrieved_context
                        source_uri = None
                        metadata = {}

                        # Check for URI attributes
                        for attr in ['uri', 'source_uri', 'url', 'source_url', 'source', 'document_uri', 'file_uri']:
                            if hasattr(context, attr):
                                source_uri = getattr(context, attr)
                                if source_uri:
                                    break

                        # Extract metadata if available
                        if hasattr(context, 'metadata'):
                            try:
                                ctx_metadata = context.metadata
                                if hasattr(ctx_metadata, 'title'):
                                    metadata['title'] = ctx_metadata.title
                                if hasattr(ctx_metadata, 'year'):
                                    metadata['year'] = ctx_metadata.year
                                if hasattr(ctx_metadata, 'category'):
                                    metadata['category'] = ctx_metadata.category
                                if hasattr(ctx_metadata, 'source_type'):
                                    metadata['source_type'] = ctx_metadata.source_type
                            except:
                                pass

                        if source_uri and isinstance(source_uri, str) and source_uri not in seen_sources:
                            seen_sources.add(source_uri)
                            sources.append({'uri': source_uri, 'metadata': metadata})

                    # Try web sources
                    if hasattr(chunk, 'web') and chunk.web:
                        for attr in ['uri', 'url']:
                            if hasattr(chunk.web, attr):
                                source_uri = getattr(chunk.web, attr)
                                if source_uri and isinstance(source_uri, str) and source_uri not in seen_sources:
                                    seen_sources.add(source_uri)
                                    sources.append({'uri': source_uri, 'metadata': {}})

            # Also check for direct source references
            if hasattr(grounding, 'retrieval_queries') and grounding.retrieval_queries:
                for query in grounding.retrieval_queries:
                    for attr in ['source_uri', 'uri', 'source']:
                        if hasattr(query, attr):
                            source_uri = getattr(query, attr)
                            if source_uri and isinstance(source_uri, str) and source_uri not in seen_sources:
                                seen_sources.add(source_uri)
                                sources.append({'uri': source_uri, 'metadata': {}})

        except Exception as e:
            # If extraction fails, return empty list
            print(f"[Debug: Error extracting sources: {str(e)}]")

        return sources

    def _search_pdf_with_google(self, pdf_title: str) -> str:
        """
        Search for a PDF document using Google Custom Search API
        [DISABLED] - Google Search API is currently inactive

        Args:
            pdf_title: Cleaned title of the PDF (e.g., "Dementia in India 2020")

        Returns:
            Empty string (Google Search API is disabled)
        """
        # Google Search API is disabled - return empty string immediately
        print("[Debug: Google Search API is disabled - returning empty string]")
        return ""

    def _find_relevant_links(self, response_text: str, num_links: int = 3) -> List[str]:
        """
        Find relevant web links based on the response content using web search

        Args:
            response_text: The response text from the bot
            num_links: Number of links to find

        Returns:
            List of relevant web URLs
        """
        links = []

        # Google Search API is disabled - skip web search
        print("[Debug: Google Search API is disabled - skipping web search]")

        # Use fallback sources only
        if len(links) < num_links:
            # Add trusted sources as fallback
            default_sources = [
                'https://www.who.int/news-room/fact-sheets/detail/dementia',
                'https://www.alz.org/alzheimers-dementia/what-is-dementia',
                'https://dementia-india.org/',
                'https://www.nimhans.ac.in/',
                'https://ardsi.org/'
            ]

            for url in default_sources:
                if url not in links and len(links) < num_links:
                    links.append(url)

        return links[:num_links]
    
    def _clean_pdf_filename(self, uri: str) -> str:
        """
        Extract and clean PDF filename for human-readable display

        Args:
            uri: The PDF URI (e.g., gs://bucket/Dementia-in-India-2020.pdf)

        Returns:
            Cleaned, human-readable title (e.g., "Dementia in India (2020)")
        """
        try:
            # Extract filename from path
            filename = uri.split('/')[-1]

            # Remove .pdf extension
            filename = filename.replace('.pdf', '').replace('.PDF', '')

            # Replace hyphens and underscores with spaces
            filename = filename.replace('-', ' ').replace('_', ' ')

            # Try to extract year if present (4 digits)
            year_match = re.search(r'\b(19|20)\d{2}\b', filename)
            year = None
            if year_match:
                year = year_match.group(0)
                # Remove year from filename
                filename = filename.replace(year, '').strip()

            # Clean up extra spaces
            filename = ' '.join(filename.split())

            # Capitalize each word
            filename = filename.title()

            # Add year back if found
            if year:
                filename = f"{filename} ({year})"

            return filename

        except Exception as e:
            print(f"[Debug: Error cleaning PDF filename: {str(e)}]")
            return uri

    def _format_citations(self, sources: List[Dict], response_text: str = "") -> str:
        """
        Format citations with metadata in human-readable format

        Args:
            sources: List of source dictionaries with 'uri' and 'metadata'
            response_text: The response text to extract keywords for web search

        Returns:
            Formatted citations section
        """
        if not sources:
            return ""

        # Separate RAG sources: web links vs PDFs/internal paths
        rag_web_links = []  # Valid web links from RAG (keep these!)
        pdf_sources = []    # PDFs that need Google Search

        for source in sources:
            uri = source.get('uri', '')
            # Check if it's a PDF or internal path (needs Google Search)
            if uri.endswith('.pdf') or '/pdf' in uri.lower() or 'gs://' in uri or not uri.startswith('http'):
                pdf_sources.append(source)
            elif uri.startswith('http://') or uri.startswith('https://'):
                # This is a valid web link from RAG - keep it!
                rag_web_links.append(source)

        print(f"[Debug: RAG web links: {len(rag_web_links)}, PDFs: {len(pdf_sources)}]")

        # Process PDFs: Use PDF file names directly (Google Search disabled)
        processed_pdfs = []
        if pdf_sources:
            print(f"[Debug: Processing {len(pdf_sources)} PDFs - using file names (Google Search disabled)]")
            for pdf_source in pdf_sources:
                uri = pdf_source.get('uri', '')
                metadata = pdf_source.get('metadata', {})

                # Get clean title from metadata or filename
                title = metadata.get('title', '') if metadata else ''
                if not title:
                    title = self._clean_pdf_filename(uri)

                # Use PDF file name directly (no Google Search)
                processed_pdfs.append({
                    'uri': uri,  # Keep original URI for reference
                    'title': title,  # Use cleaned PDF file name
                    'metadata': metadata,
                    'is_pdf': True  # Mark as PDF source
                })

        # Build final sources list: RAG web links + processed PDFs
        final_sources = []

        # Add RAG web links first (priority)
        for source in rag_web_links:
            final_sources.append({
                'uri': source.get('uri', ''),
                'title': None,
                'metadata': source.get('metadata', {})
            })

        # Add processed PDFs
        final_sources.extend(processed_pdfs)

        # If still no sources, use general web search fallback
        if not final_sources and response_text:
            print(f"[Debug: No sources from RAG, searching web as fallback]")
            found_links = self._find_relevant_links(response_text, num_links=3)
            for link in found_links:
                final_sources.append({
                    'uri': link,
                    'title': None,
                    'metadata': {}
                })

        # Limit to 5 sources
        final_sources = final_sources[:5]

        if not final_sources:
            return ""

        # Format citations with metadata
        citations_text = "\n---\n\n**References**\n\n"
        for i, source in enumerate(final_sources, 1):
            uri = source.get('uri', '')
            title = source.get('title')
            metadata = source.get('metadata', {})
            is_fallback = source.get('is_fallback', False)

            # Format citation
            is_pdf = source.get('is_pdf', False)
            if title:
                # Has a title (from PDF processing)
                citation = f"[{i}] {title}"
                year = metadata.get('year', '') if metadata else ''
                if year:
                    citation += f" ({year})"

                # For PDFs, show file name only (no URL)
                if is_pdf:
                    # Just show the PDF file name - no URL
                    citation += f"\n    (PDF Document)"
                elif not is_fallback:
                    citation += f"\n    {uri}"
                else:
                    citation += f"\n    (Link not available)"
            elif metadata and metadata.get('title'):
                # Has metadata with title (from web links)
                meta_title = metadata.get('title', '')
                year = metadata.get('year', '')
                category = metadata.get('category', '')

                citation = f"[{i}] {meta_title}"
                if year:
                    citation += f" ({year})"
                if category:
                    cat_label = category.replace('_', ' ').title()
                    citation += f" — {cat_label}"
                citation += f"\n    {uri}"
            else:
                # Just a URL
                citation = f"[{i}] {uri}"

            citations_text += citation + "\n"

        return citations_text.strip()
    
    def _check_cache_similarity(self, user_message: str) -> Tuple[bool, str]:
        """
        Check if the user message is similar to any cached queries
        Returns: (is_similar, context_to_use)
        """
        if not self.context_cache:
            return False, ""

        # Extract keywords from user message
        user_keywords = self._extract_keywords(user_message)

        if not user_keywords:
            return False, ""

        # Check similarity with cached queries
        best_match_score = 0
        best_match_context = ""

        for cached_item in self.context_cache:
            cached_keywords = self._extract_keywords(cached_item['query'])

            # Calculate overlap (Jaccard similarity)
            if cached_keywords:
                overlap = len(user_keywords & cached_keywords)
                similarity = overlap / len(user_keywords | cached_keywords)

                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_context = cached_item['response']

        # Use cached context if similarity is above threshold (30%)
        if best_match_score > 0.3:
            print("\n💾 [Using cached context - no RAG query needed]")
            return True, best_match_context

        return False, ""

    def chat(self, user_message: str) -> str:
        """
        Send a message to DIA and get a response

        Args:
            user_message: The user's question or message

        Returns:
            The assistant's response
        """
        # Check if we can use cached context
        use_cache, cached_context = self._check_cache_similarity(user_message)

        # Prepare the user message (with cached context if available)
        if use_cache:
            # Augment the message with cached context
            augmented_message = f"""Based on our previous discussion about similar topics, here's relevant context:

{cached_context}

Now, please answer this question using the context above: {user_message}"""

            # Add augmented message to conversation history
            self.conversation_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=augmented_message)]
                )
            )

            # Use config without RAG tools since we already have context
            config = self.generate_content_config_no_rag
        else:
            # Detect query category and enhance query for better retrieval
            detected_category = self._detect_query_category(user_message)
            self.current_category_filter = detected_category

            # Enhance query with category hints (workaround for SDK limitation)
            enhanced_query = self._enhance_query_with_category(user_message, detected_category)

            # Add enhanced message to conversation history for RAG retrieval
            self.conversation_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=enhanced_query)]
                )
            )

            # Use config with RAG tools
            config = self.generate_content_config

        # Generate response with streaming
        response_text = ""
        sources = []  # Store source references
        last_chunk = None

        try:
            # Stream response for display
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=self.conversation_history,
                config=config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue

                chunk_text = chunk.text
                response_text += chunk_text
                print(chunk_text, end="", flush=True)
                last_chunk = chunk  # Store last chunk for metadata extraction

            print()  # New line after response

            # Extract citations from RAG retrieval results (only if RAG was used)
            sources = []
            if not use_cache:
                try:
                    # Always make a non-streaming call to get full metadata (streaming doesn't always include it)
                    full_response = self.client.models.generate_content(
                        model=self.model,
                        contents=self.conversation_history,
                        config=config,
                    )
                    
                    # Extract sources from the response (these are from RAG)
                    sources = self._extract_sources_from_response(full_response)
                    
                    if sources:
                        print(f"\n[Debug: Found {len(sources)} RAG sources: {sources[:2]}...]")
                    else:
                        print(f"\n[Debug: No RAG sources extracted from response]")
                                    
                except Exception as e:
                    # If metadata extraction fails, continue without sources
                    print(f"\n[Note: Could not extract source references: {str(e)}]")
                    import traceback
                    traceback.print_exc()

            # Add sources to response if available
            # Insert citations before "You might also want to know" section
            if sources:
                citations_section = self._format_citations(sources, response_text)
                
                # Check if response has "You might also want to know" section
                if '**You might also want to know:**' in response_text:
                    # Split response into main content and follow-up questions
                    parts = response_text.split('**You might also want to know:**', 1)
                    main_content = parts[0].strip()
                    follow_up_section = '**You might also want to know:**' + parts[1] if len(parts) > 1 else ''
                    
                    # Insert citations between main content and follow-up questions
                    response_text = main_content + "\n\n" + citations_section
                    if follow_up_section:
                        response_text += "\n\n" + follow_up_section
                else:
                    # No follow-up section, just add citations at the end
                    response_text += "\n\n" + citations_section

            # Add assistant response to conversation history
            self.conversation_history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=response_text)]
                )
            )

            # Cache the query and response for future use (only if not using cache)
            if not use_cache:
                self.context_cache.append({
                    'query': user_message,
                    'response': response_text
                })

                # Limit cache size to last 10 queries to prevent memory bloat
                if len(self.context_cache) > 10:
                    self.context_cache.pop(0)

            return response_text

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"\n{error_msg}")

            # Provide helpful guidance for model not found errors
            if "404" in str(e) and "NOT_FOUND" in str(e):
                print("\n💡 Troubleshooting tips for Gemini 2.5 Flash:")
                print("   - The model may not be available in your region (asia-south1)")
                print("   - Try changing GOOGLE_CLOUD_LOCATION to 'us-central1' or 'global' in your .env file")
                print("   - Also update RAG_CORPUS path to match the new location")
                print("   - Or try: gemini-2.5-flash-preview-09-2025 (if available in your region)")
                print("   - Check model availability: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash")

            return error_msg

    def clear_history(self):
        """Clear the conversation history and context cache"""
        self.conversation_history = []
        self.context_cache = []

    def get_welcome_message(self) -> str:
        """
        Generate a creative welcome message that initiates the conversation
        This message should be dementia awareness related and welcoming
        """
        welcome_prompt = """Generate a warm, creative welcome message for DIA (Dementia Information Assistant). 
The message should:
- Be friendly and inviting (use 1-2 emojis thoughtfully)
- Initiate conversation about dementia awareness
- Be encouraging and supportive
- Mention that you're here to help with dementia-related questions
- Be creative but focused on dementia awareness
- Keep the main message to 3-4 sentences, warm and human-like
- Don't ask a question in the main message, but invite them to share their thoughts or questions
- IMPORTANT: At the end, include 2-3 relevant follow-up questions formatted like this:

---
**You might also want to know:**
1. [Question 1 about dementia]
2. [Question 2 about dementia]
3. [Question 3 about dementia]

Make the follow-up questions relevant to dementia awareness, symptoms, caregiving, or support. Start the message naturally, as if you're beginning a conversation."""

        try:
            # Generate welcome message using the model
            response = self.client.models.generate_content(
                model=self.model,
                contents=[welcome_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=200,
                    system_instruction=SYSTEM_PROMPT,
                )
            )
            
            welcome_text = response.text.strip()
            
            # Add to conversation history as assistant message
            self.conversation_history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=welcome_text)]
                )
            )
            
            return welcome_text
        except Exception as e:
            # Fallback welcome message if generation fails
            return """Hello! I'm DIA, your Dementia Information Assistant. 💙 I'm here to help you learn about dementia awareness, caregiving, and support resources. Whether you have questions about symptoms, care strategies, or just need someone to talk to about dementia, I'm here for you.

---
**You might also want to know:**
1. What is dementia and what are its early signs?
2. How can I support someone living with dementia?
3. What resources are available for caregivers in India?"""


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("  Welcome to DIA - Dementia Information Assistant")
    print("="*60)
    print("\nI'm here to help you learn about dementia in a friendly,")
    print("clear way. Ask me anything about dementia awareness,")
    print("symptoms, care, or support.")
    print("\n✨ New Features:")
    print("  - Smart caching: Follow-up questions use cached context")
    print("  - Suggested questions: Get relevant follow-ups after each answer")
    print("\nCommands:")
    print("  - Type your question to chat")
    print("  - Type 'clear' to start a new conversation")
    print("  - Type 'exit' or 'quit' to leave")
    print("\n" + "="*60 + "\n")


def main():
    """Main CLI loop for DIA"""

    try:
        # Initialize DIA agent
        print("Initializing DIA Agent...")
        agent = DIAAgent()
        print_welcome()

        # Generate and display welcome message from bot
        print("\n🤖 DIA: ", end="", flush=True)
        welcome_msg = agent.get_welcome_message()
        print(welcome_msg)

        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using DIA. Take care!")
                    break

                if user_input.lower() == 'clear':
                    agent.clear_history()
                    print("\n✨ Conversation history cleared. Starting fresh!")
                    continue

                # Generate and display response
                print("\n🤖 DIA: ", end="", flush=True)
                agent.chat(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using DIA. Take care!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue

    except ValueError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file with:")
        print("   - GOOGLE_CLOUD_PROJECT (your GCP project ID)")
        print("   - GOOGLE_CLOUD_LOCATION (e.g., asia-south1)")
        print("   - RAG_CORPUS (your RAG corpus path)")
        print("2. Set up Google Cloud authentication using one of:")
        print("   - Run: gcloud auth application-default login")
        print("   - Or set GOOGLE_APPLICATION_CREDENTIALS to your service account key file")
        print("\nNote: Vertex AI does NOT use API keys. It requires OAuth2 authentication.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
