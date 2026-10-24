/**
 * StudySnap AI - Centralized Feature Registry & Prompt System
 * 
 * Maps feature identifiers, visible display titles, and internal AI system instructions.
 * Prevents internal system prompts, task instructions, and templates from ever appearing
 * as visible chat messages, chat bubbles, or session history titles.
 */

export interface FeatureDefinition {
  id: string;
  visibleTitle: string;
  internalPrompt: string;
  patterns?: RegExp[];
  description?: string;
}

export const FEATURES: Record<string, FeatureDefinition> = {
  audionotes: {
    id: 'audionotes',
    visibleTitle: 'Audio to Notes',
    internalPrompt:
      'Convert the following audio transcript into structured educational notes using clean Markdown. You are an expert educational assistant that creates clear, structured study notes. Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format.',
    patterns: [
      /convert the (?:following|provided) audio transcript into structured educational notes/i,
      /convert the (?:following|provided) audio transcript/i,
    ],
    description: 'Transform lecture audio transcripts into structured, comprehensive study notes',
  },
  pyq: {
    id: 'pyq',
    visibleTitle: 'PYQ Analysis',
    internalPrompt:
      'Analyze the provided previous year question papers. Use mathematical probability and pattern recognition to identify trends, frequency of topics, and weightage. Based on this analysis, predict the most likely questions and high-yield topics for the upcoming exam.',
    patterns: [
      /analyze the provided previous year question papers/i,
      /analyze the provided previous year/i,
    ],
    description: 'Analyze previous year exam papers to identify trends and high-yield topics',
  },
  audio: {
    id: 'audio',
    visibleTitle: 'Audio Overview',
    internalPrompt:
      'Generate an engaging educational audio overview and podcast script between two hosts discussing the key concepts in the selected study materials.',
    patterns: [
      /generate an engaging educational audio overview/i,
      /create a short, engaging podcast script between two hosts/i,
    ],
    description: 'AI-generated podcast deep-dive conversation between two hosts',
  },
  mindmap: {
    id: 'mindmap',
    visibleTitle: 'Mind Map',
    internalPrompt:
      'Create a visual diagram that maps out and connects high-level concepts from the materials. Output a structured Markdown hierarchical list.',
    patterns: [
      /create a visual diagram that maps out and connects high-level concepts/i,
      /create a visual diagram that maps out/i,
    ],
    description: 'Interactive concept graph mapping out high-level relationships',
  },
  reports: {
    id: 'reports',
    visibleTitle: 'Reports',
    internalPrompt:
      'Write a structured, written brief and executive summary of the selected content. Include an Executive Summary, Key Findings, and Conclusion.',
    patterns: [
      /write a structured,? written brief and executive summary/i,
      /write a structured,? written brief/i,
    ],
    description: 'Executive summaries, academic briefings, and key findings report',
  },
  flashcards: {
    id: 'flashcards',
    visibleTitle: 'Flashcards',
    internalPrompt:
      'Create digital study flashcards extracting key terms and facts for quick memorization. Format as a Markdown table with \'Term\' and \'Definition\' columns.',
    patterns: [
      /create digital study flashcards extracting key terms/i,
      /create digital study flashcards/i,
    ],
    description: 'Bite-sized digital flashcards for active recall memorization',
  },
  quiz: {
    id: 'quiz',
    visibleTitle: 'Quiz',
    internalPrompt:
      'Generate a multiple-choice quiz to test my comprehension of the source data. Provide 5 questions with 4 options each, and an answer key at the bottom.',
    patterns: [
      /generate a multiple-choice quiz to test my comprehension/i,
      /generate a multiple-choice quiz/i,
    ],
    description: 'Interactive comprehension test with instant feedback and explanations',
  },
  slide: {
    id: 'slide',
    visibleTitle: 'Presentation Slides',
    internalPrompt:
      'Create ready-to-use presentation slides featuring AI-generated layouts and visual descriptions based on the selected sources. Format as \'Slide 1:\', \'Visual:\', \'Speaker Notes:\', etc.',
    patterns: [
      /create ready-to-use presentation slides/i,
    ],
    description: 'Structured presentation slide outlines with visual suggestions',
  },
  infographic: {
    id: 'infographic',
    visibleTitle: 'Infographic Summary',
    internalPrompt:
      'Create a visual summary script incorporating icons, statistics, and graphic layouts based on the sources. Describe the layout, colors, and text for each section of the infographic.',
    patterns: [
      /create a visual summary script incorporating icons/i,
      /create a visual summary script/i,
    ],
    description: 'Visual layout and graphics script for high-impact summaries',
  },
  table: {
    id: 'table',
    visibleTitle: 'Data Table',
    internalPrompt:
      'Extract and organize key data points from the text into a structured, exportable Markdown table.',
    patterns: [
      /extract and organize key data points from the text/i,
    ],
    description: 'Clean tabular summary organizing comparative data points',
  },
  youtube: {
    id: 'youtube',
    visibleTitle: 'YouTube to Notes',
    internalPrompt:
      'Convert the content from the provided YouTube link/video into structured educational notes using clean Markdown. You are an expert educational assistant that creates clear, structured study notes. Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format.',
    patterns: [
      /convert the content from the provided youtube link\/video into structured educational notes/i,
      /convert the content from the provided youtube/i,
    ],
    description: 'Convert video lessons into structured educational notes',
  },
  learn: {
    id: 'learn',
    visibleTitle: 'Learn & Understand',
    internalPrompt:
      'Help me learn and understand the key concepts from my study material with clear explanations, real-world examples, and step-by-step breakdowns.',
    patterns: [
      /^learn or understand something$/i,
      /help me learn and understand the key concepts/i,
    ],
    description: 'Step-by-step guided breakdown of concepts with real-world examples',
  },
};

const RESEARCH_REGEX = /^Perform a (Quick Summary|Deep Research|Fact Check|Comprehensive Analysis) on .* for: "(.*)"\.?.*$/i;

/**
 * Identify if given text matches a known feature's prompt or signature.
 */
export function detectFeature(text?: string | null): FeatureDefinition | null {
  if (!text) return null;
  const normalized = text.trim();

  // Check research mode query
  const researchMatch = normalized.match(RESEARCH_REGEX);
  if (researchMatch) {
    const mode = researchMatch[1];
    const query = researchMatch[2];
    return {
      id: 'research',
      visibleTitle: query ? `${mode}: ${query}` : mode,
      internalPrompt: normalized,
    };
  }

  for (const key of Object.keys(FEATURES)) {
    const feat = FEATURES[key];
    if (normalized === feat.internalPrompt || normalized.startsWith(feat.internalPrompt.substring(0, 50))) {
      return feat;
    }
    if (feat.patterns) {
      for (const pattern of feat.patterns) {
        if (pattern.test(normalized)) {
          return feat;
        }
      }
    }
  }

  return null;
}

/**
 * Convert any raw system prompt or leaked instruction into its user-friendly display title.
 * If text is an ordinary user message, it is returned intact.
 */
export function getFeatureDisplayTitle(contentOrTitle?: string | null): string {
  if (!contentOrTitle) return 'New Chat';
  const detected = detectFeature(contentOrTitle);
  if (detected) {
    return detected.visibleTitle;
  }
  return contentOrTitle;
}

/**
 * Checks if a given text is an internal system prompt.
 */
export function isInternalSystemPrompt(text?: string | null): boolean {
  if (!text) return false;
  return detectFeature(text) !== null;
}

/**
 * Get clean session title (sanitizes legacy prompt strings and limits length).
 */
export function getCleanSessionTitle(rawTitle?: string | null): string {
  if (!rawTitle) return 'New Chat';
  const detected = detectFeature(rawTitle);
  if (detected) {
    return detected.visibleTitle;
  }
  const clean = rawTitle.trim();
  if (clean.length > 32) {
    return clean.substring(0, 30) + '...';
  }
  return clean;
}
