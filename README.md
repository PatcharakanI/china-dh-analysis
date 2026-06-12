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

1. Push this folder to a GitHub repository.
2. Run:

```bash
npm install
npm run build
```

3. Deploy the `dist/` folder to GitHub Pages.

One simple option is to use the `gh-pages` package:

```bash
npm install --save-dev gh-pages
```

Then add these scripts to `package.json` if you want command-line deployment:

```json
{
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d dist"
  }
}
```

Then run:

```bash
npm run deploy
```

The Vite config uses:

```js
base: "./"
```

This makes the site work as a static GitHub Pages deployment without a backend.

## Method Note

The website displays dictionary-based framing scores, co-occurrence measures, and embedding similarity indicators. These are not objective proof of sentiment, intention, or causality. They are transparent textual indicators meant to support close reading and historical interpretation.

## Main Sections

- Landing / introduction
- Timeline dashboard
- Economic securitization
- Semantic shift
- Framing by section and news desk
- Keywords and close-reading article viewer
- Method and limitations
