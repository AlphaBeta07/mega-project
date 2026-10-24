"""
StudySnap AI - Centralized Feature & Prompt Registry (Backend)
Provides mapping between feature identifiers, user-facing display titles,
and internal AI system prompts/instructions.
Ensures internal system prompts never leak into user-facing chat messages or session titles.
"""
from typing import Optional, Dict, Any, Tuple
import re

# Centralized feature definitions
FEATURE_DEFINITIONS = {
    "audionotes": {
        "id": "audionotes",
        "visible_title": "Audio to Notes",
        "internal_prompt": (
            "Convert the following audio transcript into structured educational notes using clean Markdown. "
            "You are an expert educational assistant that creates clear, structured study notes. "
            "Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format."
        ),
        "patterns": [
            r"convert the (?:following|provided) audio transcript into structured educational notes",
            r"convert the (?:following|provided) audio transcript",
        ]
    },
    "pyq": {
        "id": "pyq",
        "visible_title": "PYQ Analysis",
        "internal_prompt": (
            "Analyze the provided previous year question papers. Use mathematical probability and pattern recognition to identify trends, "
            "frequency of topics, and weightage. Based on this analysis, predict the most likely questions and high-yield topics for the upcoming exam."
        ),
        "patterns": [
            r"analyze the provided previous year question papers",
            r"analyze the provided previous year",
        ]
    },
    "audio": {
        "id": "audio",
        "visible_title": "Audio Overview",
        "internal_prompt": (
            "Generate an engaging educational audio overview and podcast script based on the selected study materials."
        ),
        "patterns": [
            r"generate an engaging educational audio overview",
            r"create a short, engaging podcast script between two hosts",
        ]
    },
    "mindmap": {
        "id": "mindmap",
        "visible_title": "Mind Map",
        "internal_prompt": (
            "Create a visual diagram that maps out and connects high-level concepts from the materials. Output a structured Markdown hierarchical list."
        ),
        "patterns": [
            r"create a visual diagram that maps out and connects high-level concepts",
            r"create a visual diagram that maps out",
        ]
    },
    "reports": {
        "id": "reports",
        "visible_title": "Reports",
        "internal_prompt": (
            "Write a structured, written brief and executive summary of the selected content. Include an Executive Summary, Key Findings, and Conclusion."
        ),
        "patterns": [
            r"write a structured,? written brief and executive summary",
            r"write a structured,? written brief",
        ]
    },
    "flashcards": {
        "id": "flashcards",
        "visible_title": "Flashcards",
        "internal_prompt": (
            "Create digital study flashcards extracting key terms and facts for quick memorization. Format as a Markdown table with 'Term' and 'Definition' columns."
        ),
        "patterns": [
            r"create digital study flashcards extracting key terms",
            r"create digital study flashcards",
        ]
    },
    "quiz": {
        "id": "quiz",
        "visible_title": "Quiz",
        "internal_prompt": (
            "Generate a multiple-choice quiz to test my comprehension of the source data. Provide 5 questions with 4 options each, and an answer key at the bottom."
        ),
        "patterns": [
            r"generate a multiple-choice quiz to test my comprehension",
            r"generate a multiple-choice quiz",
        ]
    },
    "slide": {
        "id": "slide",
        "visible_title": "Presentation Slides",
        "internal_prompt": (
            "Create ready-to-use presentation slides featuring AI-generated layouts and visual descriptions based on the selected sources. Format as 'Slide 1:', 'Visual:', 'Speaker Notes:', etc."
        ),
        "patterns": [
            r"create ready-to-use presentation slides",
        ]
    },
    "infographic": {
        "id": "infographic",
        "visible_title": "Infographic Summary",
        "internal_prompt": (
            "Create a visual summary script incorporating icons, statistics, and graphic layouts based on the sources. Describe the layout, colors, and text for each section of the infographic."
        ),
        "patterns": [
            r"create a visual summary script incorporating icons",
            r"create a visual summary script",
        ]
    },
    "table": {
        "id": "table",
        "visible_title": "Data Table",
        "internal_prompt": (
            "Extract and organize key data points from the text into a structured, exportable Markdown table."
        ),
        "patterns": [
            r"extract and organize key data points from the text",
        ]
    },
    "youtube": {
        "id": "youtube",
        "visible_title": "YouTube to Notes",
        "internal_prompt": (
            "Convert the content from the provided YouTube link/video into structured educational notes using clean Markdown. "
            "You are an expert educational assistant that creates clear, structured study notes. "
            "Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format."
        ),
        "patterns": [
            r"convert the content from the provided youtube link/video into structured educational notes",
            r"convert the content from the provided youtube",
        ]
    },
    "learn": {
        "id": "learn",
        "visible_title": "Learn & Understand",
        "internal_prompt": (
            "Help me learn and understand the key concepts from my study material with clear explanations, real-world examples, and step-by-step breakdowns."
        ),
        "patterns": [
            r"^learn or understand something$",
            r"help me learn and understand the key concepts",
        ]
    }
}

RESEARCH_PATTERN = re.compile(r'^Perform a (Quick Summary|Deep Research|Fact Check|Comprehensive Analysis) on .* for: "(.*)"\.?.*$', re.IGNORECASE)

def detect_feature(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Detect if given text matches any known feature prompt or signature."""
    if not text:
        return None
    normalized = text.strip()
    
    # Check research pattern
    m = RESEARCH_PATTERN.match(normalized)
    if m:
        mode = m.group(1)
        query = m.group(2)
        return {
            "id": "research",
            "visible_title": f"{mode}: {query}" if query else f"{mode}",
            "internal_prompt": normalized
        }

    for feat_id, feat in FEATURE_DEFINITIONS.items():
        # Exact match or starts with internal prompt
        if normalized == feat["internal_prompt"] or normalized.startswith(feat["internal_prompt"][:60]):
            return feat
        # Check patterns
        for p in feat.get("patterns", []):
            if re.search(p, normalized, re.IGNORECASE):
                return feat
                
    return None

def sanitize_feature_title(text: Optional[str]) -> str:
    """
    Return clean user-facing title.
    If the text contains an internal system prompt or feature instruction,
    replaces it with the clean feature title (e.g. 'Audio to Notes').
    """
    if not text:
        return "New Chat"
    feat = detect_feature(text)
    if feat:
        return feat["visible_title"]
    
    # For regular messages, clean and truncate if too long
    clean = text.strip()
    if len(clean) > 40:
        return clean[:37] + "..."
    return clean

def sanitize_message_content(content: Optional[str]) -> str:
    """
    If a message content is an internal system prompt, return its clean display title.
    Otherwise return content unmodified.
    """
    if not content:
        return ""
    feat = detect_feature(content)
    if feat:
        return feat["visible_title"]
    return content

def resolve_request_instruction(message: str, system_instruction: Optional[str] = None, feature: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    """
    Resolve (query_for_model, system_instruction, display_title).
    Handles cases where client sent prompt in message or in system_instruction.
    """
    display_title = message
    resolved_instruction = system_instruction
    
    feat = None
    if feature and feature in FEATURE_DEFINITIONS:
        feat = FEATURE_DEFINITIONS[feature]
    else:
        feat = detect_feature(message)
        
    if feat:
        display_title = feat["visible_title"]
        if not resolved_instruction:
            resolved_instruction = feat["internal_prompt"]
        # The user query sent to the LLM should be the clean feature title,
        # while the internal prompt is sent in system_instruction
        return display_title, resolved_instruction, display_title
    
    return message, resolved_instruction, sanitize_feature_title(message)
