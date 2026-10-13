# Baseten ideal customer profile (ICP)

fit.py reads this file and gives it to Claude when scoring each company. Edit it to change what counts as each kind of evidence. The order in which segments are decided lives in fit.py (SEGMENT_RULES), so it stays the same from run to run.

## Tier 1: AI-native scale-ups (main focus)

### Who they are

* Series B–D AI application companies, roughly 50–500 employees, with real production traffic
* Verticals where Baseten has proof: healthcare and clinical AI, coding and developer tools, GTM and data enrichment, legal, voice/video/creative
* Inference is a core cost of goods sold, so every user action triggers model calls

### What they're running

* Fine-tuned or open-source models in production, or actively moving there
* Self-managed GPU or vLLM setups that are getting painful, or a competitor like Together, Fireworks, or Modal
* Large closed-API bills they want to bring down

### Their pain

* Gross margins hurt by compute costs
* Latency limiting the product (voice, real-time, agents)
* Reliability problems at scale, like rate limits and outages
* Infra engineers spending time on serving instead of on the product

### Who to talk to

* Champion: Head of ML or ML infra lead
* Buyer: CTO / VP Engineering, often a technical co-founder

### When to reach out

* Just raised a round
* Hiring ML infra engineers
* Launching latency-sensitive features
* Publicly discussing AI costs or margins

Why first: These companies look like Baseten's best customers, the value is easy to show (cost per inference, latency, uptime), and accounts grow automatically as usage grows.

## Tier 2: Enterprises and regulated verticals (bigger deals, slower)

### Who they are

* Large SaaS companies adding AI features, like Notion
* Financial services, insurance, healthcare systems, and legal firms building internal or customer-facing AI
* Vertical AI companies selling into those industries, like Rogo selling to banks
* Classify AI vendors by their own vertical, not their customers' industry: healthcare AI vendors (including ones selling to insurers) are tier 1 or tier 3 by stage

### What they care about

* Data control, security, compliance
* Hybrid or self-hosted deployment
* Running in their own cloud

### Who to talk to

* VP Engineering or Head of AI Platform, plus security and procurement

How to get in: Lead with security, IP protection, and multi-cloud flexibility. Use the AWS partnership so they can buy through the AWS Marketplace and count it against existing cloud spend, which removes a big procurement obstacle.

Why second: Deal sizes are larger, but sales cycles run 6–12 months, so these deals need the Tier 1 customer stories first.

## Tier 3: Early-stage AI startups (self-serve and nurture)

### Who they are

* Seed to Series A teams building on open models

### How to sell to them

* Mostly product-led: they sign up for Model APIs and pay for usage
* SDRs step in only when usage spikes or they raise money

Why: Low cost to acquire, and some will become the next Cursor. Win them early, then expand.

## Who to deprioritize

* Frontier labs that build their own serving stack
* Companies only experimenting with AI at low volume
* Teams happy on closed APIs with no plans to use their own models
