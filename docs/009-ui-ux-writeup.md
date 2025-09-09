# website

help me write a Plan in a formal what for my objective I want to create a web application where
front end controlled by simple RABC, using Nuxt.js versio n 4 backend is Python,

## UI Theme

https://ui.nuxt.com/ Create beautiful, responsive & accessible web apps quickly with Vue or Nuxt.
Nuxt UI is an open-source UI library of 50+ customizable components built with Tailwind CSS and Reka
UI.

leaning into a sleek “ink/indigo” look with Nuxt UI + Reka UI (Tailwind underneath), here’s a
cohesive theme you can drop in that keeps your primary as your custom black scale, adds sensible
semantic colors, and tunes common components (buttons, inputs, cards, modals, tooltips) for great
a11y and polish.

⸻

1. Tokens with @theme (light + dark)

Keep your black scale and add semantic tokens + radii, shadows, and rings Nuxt UI can benefit from.
Override a few values in .dark to keep contrast strong.

```css
/* app/assets/css/theme.css */
@theme {
  /* Neutral “black” scale (yours) */
  --color-black-50: #f7f7f7;
  --color-black-100: #e3e3e3;
  --color-black-200: #c8c8c8;
  --color-black-300: #a4a4a4;
  --color-black-400: #818181;
  --color-black-500: #666666;
  --color-black-600: #515151;
  --color-black-700: #434343;
  --color-black-800: #383838;
  --color-black-900: #313131;
  --color-black-950: #1a1a1a;

  /* Semantic aliases */
  --color-primary-50: var(--color-black-50);
  --color-primary-100: var(--color-black-100);
  --color-primary-200: var(--color-black-200);
  --color-primary-300: var(--color-black-300);
  --color-primary-400: var(--color-black-400);
  --color-primary-500: var(--color-black-700); /* darker for a11y */
  --color-primary-600: var(--color-black-800);
  --color-primary-700: var(--color-black-900);
  --color-primary-800: var(--color-black-950);
  --color-primary-900: #0f0f0f;

  /* Secondary & Status (use Tailwind defaults later, but expose tokens too) */
  --color-secondary-500: #6366f1; /* indigo-500 */
  --color-success-500: #10b981; /* emerald-500 */
  --color-warning-500: #f59e0b; /* amber-500 */
  --color-error-500: #f97316; /* orange-500 */
  --color-info-500: #0ea5e9; /* sky-500 */

  /* Surfaces & content */
  --ui-bg: #ffffff;
  --ui-muted: #f8f8f8;
  --ui-card: #ffffff;
  --ui-border: #e5e7eb; /* zinc-200 */
  --ui-foreground: #111827; /* zinc-900 */

  /* Radius / shadow / ring */
  --ui-radius-sm: 0.375rem;
  --ui-radius-md: 0.5rem;
  --ui-radius-lg: 0.75rem;
  --ui-shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --ui-shadow-md: 0 8px 24px -8px rgb(0 0 0 / 0.15);
  --ui-ring: 2px;
  --ui-ring-color: color-mix(in oklab, var(--color-primary-500) 45%, white);
}

/* Dark mode refinements */
.dark {
  --ui-bg: #0b0b0b;
  --ui-muted: #0f0f10;
  --ui-card: #111213;
  --ui-border: #2b2b2b;
  --ui-foreground: #e5e7eb; /* zinc-200 */

  /* Slightly lift primary for dark to preserve focus rings/hover */
  --color-primary-500: var(--color-black-600);
  --color-primary-600: var(--color-black-700);
  --color-primary-700: var(--color-black-800);
  --ui-ring-color: color-mix(in oklab, var(--color-primary-500) 70%, black);
}
```

2. Tailwind config: wire CSS variables to utilities

Expose your black palette to Tailwind so bg-black-700 etc. map to your tokens. Also ensure dark mode
is class-based (Nuxt UI default).

```css

// tailwind.config.ts
import type { Config } from 'tailwindcss'

export default <Partial<Config>>{
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        black: {
          50:  'var(--color-black-50)',
          100: 'var(--color-black-100)',
          200: 'var(--color-black-200)',
          300: 'var(--color-black-300)',
          400: 'var(--color-black-400)',
          500: 'var(--color-black-500)',
          600: 'var(--color-black-600)',
          700: 'var(--color-black-700)',
          800: 'var(--color-black-800)',
          900: 'var(--color-black-900)',
          950:'var(--color-black-950)',
          DEFAULT: 'var(--color-black-700)'
        }
      },
      borderRadius: {
        DEFAULT: 'var(--ui-radius-md)',
        md: 'var(--ui-radius-md)',
        lg: 'var(--ui-radius-lg)',
        sm: 'var(--ui-radius-sm)',
      },
      boxShadow: {
        'sm': 'var(--ui-shadow-sm)',
        'md': 'var(--ui-shadow-md)',
      }
    }
  },
  plugins: []
}

```

3. Nuxt UI app.config.ts: colors + component polish

This keeps your primary: "black" and adds consistent variants with strong focus states and
accessible contrasts.

```css
// app.config.ts
export default defineAppConfig({
  ui: {
    // Palette mapping to Tailwind color families
    colors: {
      primary: 'black',
      secondary: 'indigo',
      gray: 'zinc',
      success: 'emerald',
      warning: 'amber',
      error: 'orange',
      info: 'sky'
    },

    // Global component defaults
    strategy: 'class',
    focusRing: 'ring-2 ring-offset-2 ring-offset-white dark:ring-offset-black',
    rounded: 'rounded-md',

    // Buttons
    button: {
      default: {
        size: 'md',
        color: 'primary',
        variant: 'solid',
        class: 'font-medium transition-shadow focus-visible:outline-none'
      },
      size: {
        sm: 'h-9 px-3 text-sm',
        md: 'h-10 px-4 text-sm',
        lg: 'h-11 px-5 text-base'
      },
      variant: {
        solid:
          // Use your ink palette for solid; ensure text contrast
          'bg-black-700 text-white hover:bg-black-800 active:bg-black-900 ' +
          'focus-visible:ring-primary-500 focus-visible:ring-2',
        soft:
          // soft/“subtle” background for secondary actions
          'bg-black-100 text-black-900 hover:bg-black-200 dark:bg-black-800 dark:text-zinc-100 dark:hover:bg-black-700 ' +
          'focus-visible:ring-primary-500 focus-visible:ring-2',
        outline:
          'border border-black-300 text-black-900 hover:bg-black-50 dark:border-black-700 dark:text-zinc-100 dark:hover:bg-black-800 ' +
          'focus-visible:ring-primary-500 focus-visible:ring-2',
        ghost:
          'text-black-900 hover:bg-black-50 dark:text-zinc-100 dark:hover:bg-black-800 ' +
          'focus-visible:ring-primary-500 focus-visible:ring-2',
        link:
          'text-indigo-600 hover:text-indigo-700 underline-offset-4 hover:underline ' +
          'focus-visible:ring-primary-500 focus-visible:ring-2'
      }
    },

    // Inputs (UInput, UTextarea, USelect)
    input: {
      base:
        'bg-[var(--ui-card)] text-[var(--ui-foreground)] ' +
        'placeholder:text-zinc-400 dark:placeholder:text-zinc-500 ' +
        'border border-[var(--ui-border)] focus:border-black-500 ' +
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500',
      rounded: 'rounded-md',
      color: {
        primary: 'focus:border-black-500',
        error: 'border-orange-400 focus:ring-orange-500'
      }
    },

    // Cards (UCard)
    card: {
      base:
        'bg-[var(--ui-card)] text-[var(--ui-foreground)] ' +
        'border border-[var(--ui-border)] shadow-sm',
      rounded: 'rounded-lg'
    },

    // Modals/Popovers (UModal, UPopover)
    modal: {
      overlay: 'bg-black/50 backdrop-blur-sm',
      container: 'p-4',
      base:
        'bg-[var(--ui-card)] text-[var(--ui-foreground)] ' +
        'border border-[var(--ui-border)] shadow-md rounded-lg'
    },

    // Tooltip
    tooltip: {
      base:
        'bg-black-900 text-white px-2 py-1 rounded ' +
        'shadow-sm ring-1 ring-black-800'
    },

    // Badges
    badge: {
      variant: {
        solid: 'bg-black-800 text-white',
        soft:  'bg-black-100 text-black-900 dark:bg-black-800 dark:text-zinc-100',
        outline: 'border border-black-300 text-black-900 dark:border-black-700 dark:text-zinc-100'
      }
    },
  }
})
```

• Contrast-first: solid buttons use black-700+ with white text for WCAG contrast; soft/outline
variants keep subtle surfaces but strong focus rings. • Consistent focus:
focus-visible:ring-primary-500 everywhere makes keyboard navigation obvious. • Surface tokens:
components read from --ui-card, --ui-border, so light/dark flips are painless. • Status colors: use
Tailwind’s well-known scales (emerald/amber/orange/sky) while keeping your ink primary.

⸻

4. Quick usage examples

```vue
<script setup lang="ts"></script>

<template>
  <div class="space-y-4">
    <UButton>Primary</UButton>
    <UButton variant="soft">Soft</UButton>
    <UButton variant="outline">Outline</UButton>
    <UButton variant="ghost">Ghost</UButton>
    <UButton variant="link" color="secondary">Learn more</UButton>

    <UInput placeholder="Search…" icon="i-heroicons-magnifying-glass-20-solid" />
    <UCard>
      <template #header>Card title</template>
      Content
      <template #footer>
        <UButton size="sm">Action</UButton>
      </template>
    </UCard>
  </div>
</template>
```

5. Optional niceties

- Pressed/active states: add active:translate-y-[1px] to buttons for a tactile feel.
- Transitions: add transition-colors duration-150 to interactive elements.
- Focus ring offset: on dark surfaces, keep ring-offset-black for better visibility (already in
  focusRing).
- Density: if you want a compact look, change button/input sizes to h-9 / px-3 across the board.

## User Register of login

on the front side we have a user comes in either a new user or already registered user comes in if
new user can register, if already registerd can log in

### Forgot Password

Email send temp password via email

## Navigation

### Top Navigation :

- **The top navigation bar** should contain the most critical, high-level pages and tools that users
  need no matter where they are on the site.
- **Search Bar**: A crucial feature for content-heavy sites that helps users find specific
  information quickly.
- **User Profile and Actions**: Icons or links for user-specific functions, typically on the right
  side of the navigation
- **Profile / Account**
  - **Settings**
  - **Notifications**

### Left Navigation :

Standard components for side navigation:

- **Detailed Menu Structure**: A comprehensive list of sections, often with expandable, multi-level
  sub-menus for deeper navigation.
- **Contextual Actions**: Features or options that relate specifically to the current task or page,
  keeping users in the same "workspace".
- **System Controls**: Administrative links and user account management features.
- **System Controls**: Administrative links and user account management features.
- **Feature Groupings**: Related menu items can be visually grouped with headers to improve
  readability and information hierarchy.
- **Collapsible Design**: The ability to collapse the sidebar (often to a set of icons) to maximize
  screen space for the main content area.
- **Icons with Text Labels:** Using icons alongside text labels in the sidebar improves the
  scannability of navigation items

## Page Flow

### Start

- Dashboard
- list of Existing PRD's
- Scorecard of the PRD's
- Pending
- Completed
- Link to Create a New PRD

### New PRD

-- Start with a webform, Ask the user to type Of course. As an Expert UX/UI Engineer specializing in
AI-powered tools for Project Managers, I believe the ideal workflow is less of a form and more of a
**conversational, collaborative partnership** with an AI agent. The goal is to leverage the LLM's
breadth of knowledge and structuring capabilities while keeping the human (the PM) firmly in the
driver's seat as the editor, approver, and ultimate decision-maker.

Here is the ideal, sequential, agentic workflow I would design.

---

### **Project Genesis AI Agent: The Ideal Workflow**

**Core Philosophy:** The AI is a meticulous, junior business analyst or project strategist. Its
primary job is to **ask brilliant, clarifying questions** before it ever attempts to produce a final
deliverable. It assumes nothing and seeks to understand the user's intent deeply.

**UI Principle:** The interface should feel like a focused, professional conversation. The main
canvas is a structured chat interface, but with rich text editing, approval controls, and a
persistent "project spine" visible on the side, growing as the user progresses.

---

### **Phase 0: The Invitation**

**User Action:** Lands on the webapp. The interface is clean, with a central input field. **UI
Prompt:** "Welcome. Do you have a project idea? Describe it in a sentence or a paragraph." **Input
Field:** Large, multi-line text area with a placeholder: "e.g., 'Build a new mobile app for our
loyalty program' or 'Migrate our internal CRM from Salesforce to HubSpot'."

---

### **Phase 1: The Core Objective & Initial Clarification Loop**

1.  **User Action:** Types initial idea (e.g., "Build a new mobile app for our loyalty program") and
    hits `Enter`.
2.  **AI Action:**
    - The LLM first analyzes the input for ambiguity. It doesn't generate an objective yet.
    - It generates **3-5 high-impact, clarifying questions** based on standard project charter
      elements.
    - **Example Questions:**
      - "What is the primary business problem this new app will solve? (e.g., low customer
        retention, low redemption rates)"
      - "Who is the target audience for this app? (e.g., existing loyalty members, new younger
        demographics)"
      - "Do you have any known technical constraints? (e.g., must integrate with our existing Oracle
        database, must be built with React Native)"
      - "What does 'success' look like for this project 6 months after launch? (e.g., 20% increase
        in repeat purchases, 50k app downloads)"
3.  **UI Presentation:** The questions appear in a distinct, slightly indented section titled "To
    ensure we start correctly, I need a bit more context:". Each question has its own input field.
4.  **User Action:** Answers the questions. They can choose to answer all or just some.

_(This clarification loop can be iterative. The AI can ask follow-up questions based on the user's
answers if it detects new ambiguities.)_

---

### **Phase 2: Drafting & Approving the Project Objective**

1.  **AI Action:** Using the initial idea + all Q&A context, the LLM now generates a
    **well-structured Project Objective Statement**. It will be SMART (Specific, Measurable,
    Achievable, Relevant, Time-bound) by default.
    - **Example Output:** "**Objective:** To design and launch a native mobile application for the
      'AlphaRewards' loyalty program by Q3 2024, targeting existing members. The primary goal is to
      increase customer retention by 15% and point redemption rates by 25% within the first year
      post-launch by providing a seamless user experience for checking points, receiving
      personalized offers, and redeeming rewards in-store."
2.  **UI Presentation:**
    - The objective is displayed in a rich text box with editing tools (bold, italic, etc.).
    - Two prominent buttons: **`Edit & Refine`** and **`Accept & Continue`**.
3.  **User Action:**
    - They can directly edit the text in the box.
    - Clicking **`Edit & Refine`** could allow them to type instructions like "make the timeline
      more aggressive" or "emphasize cost-saving more." The AI would re-draft based on this
      feedback.
    - Clicking **`Accept & Continue`** locks this section and adds it to the growing "Project
      Charter" outline on the left-hand panel.

---

### **Phase 3: Section-by-Section Co-Creation**

The AI now understands this is a formal charter. It will systematically build it out, one section at
a time, following the same **Clarify -> Draft -> Edit -> Approve** pattern for each major component.

**The next prompt is never "What's next?".** The AI drives the process:

**AI Prompt:** "Great. Now that we have our objective, let's define the project scope. I can draft
an initial version. To do this well, I should ask a few questions to understand what's explicitly in
and out of scope." _(AI asks 2-3 scope-related questions, e.g., "Will this project include
developing the backend API, or are we only building the front-end mobile client?")_

The process repeats for each core section:

1.  **Scope (In-Scope / Out-of-Scope)**
2.  **Key Deliverables** (e.g., "High-fidelity UI prototypes", "TestFlight build", "App Store
    release")
3.  **High-Level Timeline & Milestones** (The AI will propose phases: Discovery, Design,
    Development, Testing, Launch)
4.  **Stakeholder Identification** (The AI will ask: "Who are the executive sponsors? Who are the
    key business leads from Marketing and Operations?")
5.  **High-Level Budget & Resources** (The AI will ask about rough budget ranges or team size)
6.  **Success Metrics / KPIs** (Building on the Objective)
7.  **Key Assumptions & Risks** (The AI will proactively list common risks for a mobile app
    project - e.g., "App Store approval delays", "Changing privacy regulations" - and ask the user
    to validate or add others.)

---

### **Phase 4: Synthesis & Finalization**

1.  **AI Action:** After all sections are approved by the user, the AI agent synthesizes everything
    into a single, beautifully formatted document (Project Charter).
2.  **UI Presentation:**
    - A "Document View" tab appears, showing the complete charter in a clean, professional template.
    - Options to **`Export as PDF`**, **`Export to Word`**, or **`Copy to Clipboard`** are
      prominent.
    - A final input field appears: **"What would you like to do next?"** with suggestive buttons:
      - `Create a PRD (Product Requirements Document)`: This would be the next agentic workflow,
        using the charter as its foundational context.
      - `Generate a Project Timeline in Gantt Format`: Triggers a different agent to break
        milestones into tasks.
      - `Share with Stakeholders`: Generates a summary email for review.

---

### **Key UX/UI & Technical Considerations:**

- **Persistent Context:** The entire conversation history and all approved text form the context
  window for the LLM. This is non-negotiable for coherence.
- **The "Project Spine":** A left-hand sidebar always shows the status of each section (e.g., ⏳
  `Drafting`, ✏️ `Needs Review`, ✅ `Approved`). This provides a constant sense of progress.
- **User Control:** The user can **go back** to any previously approved section and edit it. This
  change would then be re-contextualized for the AI, which might ask if subsequent sections need
  updating (e.g., "You've increased the project scope. Should we revisit the timeline and budget
  sections?").
- **Transparency:** The AI should subtly explain _why_ it's asking a question (e.g., "I'm asking
  about the backend because it will majorly impact the scope and timeline."). This builds trust and
  teaches the user to think like a seasoned PM.
- **Agentic Memory:** The AI should remember the user's preferences across sessions (e.g., "Last
  time you preferred more aggressive timelines, should I factor that in?").

This workflow transforms the daunting task of starting a project from a blank page into a guided,
structured, and efficient dialogue, empowering Project Managers to leverage AI as a powerful
co-pilot from the very first second.
