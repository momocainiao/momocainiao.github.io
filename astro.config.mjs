import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://momocainiao.github.io',
  base: '/',
    markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [rehypeKatex, { output: 'mathml' }] // 强制只输出 HTML，不输出 MathML
    ],
  },
});
