import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/overview">
            Read the Docs
          </Link>
          <Link
            className="button button--secondary button--lg"
            style={{marginLeft: '1rem'}}
            href="https://github.com/Qomplement/claudeSeek">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

function Feature({title, description}: {title: string; description: string}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md padding-vert--lg">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="DeepSeek-OCR-2 + vLLM compatibility, integration, and deployment">
      <HomepageHeader />
      <main>
        <section className="container margin-vert--xl">
          <div className="row">
            <Feature
              title="Architecture Deep Dive"
              description="Visual Causal Flow, SAM ViT-B + Qwen2-0.5B encoder, hybrid attention masks, and the full 3B MoE pipeline."
            />
            <Feature
              title="vLLM Integration"
              description="Model registry, out-of-tree registration, AdapterLogitsProcessor migration, and the v0 to v1 engine changes."
            />
            <Feature
              title="Ready to Deploy"
              description="Docker, API server, batch inference examples. Works with vLLM from source or pinned 0.8.5."
            />
          </div>
        </section>
      </main>
    </Layout>
  );
}
