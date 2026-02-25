import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'DeepSeek-OCR-2 + vLLM',
  tagline: 'Making DeepSeek-OCR-2 run on the latest vLLM inference engine',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://qomplement.github.io',
  baseUrl: '/claudeSeek/',

  organizationName: 'Qomplement',
  projectName: 'claudeSeek',

  onBrokenLinks: 'warn',

  markdown: {
    format: 'md',
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/Qomplement/claudeSeek/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'DeepSeek-OCR-2 + vLLM',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/Qomplement/claudeSeek',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            { label: 'Overview', to: '/docs/overview' },
            { label: 'Architecture', to: '/docs/architecture' },
            { label: 'Deployment', to: '/docs/deployment' },
          ],
        },
        {
          title: 'Resources',
          items: [
            { label: 'DeepSeek-OCR-2 (HuggingFace)', href: 'https://huggingface.co/deepseek-ai/DeepSeek-OCR-2' },
            { label: 'vLLM Docs', href: 'https://docs.vllm.ai/en/latest/' },
            { label: 'DeepSeek-OCR-2 GitHub', href: 'https://github.com/deepseek-ai/DeepSeek-OCR-2' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'GitHub', href: 'https://github.com/Qomplement/claudeSeek' },
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} Qomplement. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'docker'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
