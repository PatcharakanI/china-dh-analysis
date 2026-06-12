# NYT China Discourse Digital History Website

This is a static Vite + React website for the New York Times China discourse project.

The site uses local JSON data from:

```text
public/data/
```

The data was generated from the balanced analysis outputs in:

```text
result_securitization_balanced/
```

## Local Development

Install dependencies:

```bash
npm install
```

Run the local development server:

```bash
npm run dev
```

Build the static site:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## GitHub Pages Deployment

The Vite config uses:

```js
base: "./"
```

This makes the site work as a static GitHub Pages deployment without a backend.

Deploy with:

```bash
npm run build
npm run deploy
```

Then set GitHub Pages to deploy from the `gh-pages` branch.

## Method Note

The website displays dictionary-based framing scores, co-occurrence measures, and embedding similarity indicators. These are not objective proof of sentiment, intention, or causality. They are transparent textual indicators meant to support close reading and historical interpretation.

## Main Sections

- Exhibit
- About
- Terminology
- Result
- Close Reading
