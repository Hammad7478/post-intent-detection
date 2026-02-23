const fs = require('fs');
const path = require('path');

const annotatedDir = path.join(__dirname, 'data', 'annotated');
const sourceFiles = [
  'cscareerquestions_modelA.json',
  'personalfinance_modelA.json',
  'careerguidance_modelA.json',
];

const LABEL_TO_FILENAME = {
  ADVICE_SEEKING: 'advice_seeking_modelA.json',
  PERSONAL_EXPERIENCE: 'personal_experience_modelA.json',
  OPINION: 'opinion_modelA.json',
  OTHER: 'other_modelA.json',
};

function main() {
  const seenIds = new Set();
  const allPosts = [];

  for (const file of sourceFiles) {
    const filePath = path.join(annotatedDir, file);
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    for (const post of data) {
      if (seenIds.has(post.id)) continue;
      seenIds.add(post.id);
      allPosts.push(post);
    }
  }

  const byLabel = {
    ADVICE_SEEKING: [],
    PERSONAL_EXPERIENCE: [],
    OPINION: [],
    OTHER: [],
  };

  for (const post of allPosts) {
    const label = post.label;
    if (!byLabel[label]) byLabel[label] = [];
    byLabel[label].push(post);
  }

  for (const [label, filename] of Object.entries(LABEL_TO_FILENAME)) {
    const posts = byLabel[label] || [];
    const outPath = path.join(annotatedDir, filename);
    fs.writeFileSync(outPath, JSON.stringify(posts, null, 2), 'utf8');
    console.log(filename, posts.length, 'posts');
  }

  const totalOut = Object.values(byLabel).reduce((s, arr) => s + arr.length, 0);
  console.log('\nTotal posts processed:', allPosts.length);
  console.log('Total in label files:', totalOut);

  if (totalOut !== allPosts.length) {
    console.error('MISMATCH: total in files !== total posts');
    process.exit(1);
  }

  for (const [label, posts] of Object.entries(byLabel)) {
    const ids = posts.map(p => p.id);
    const unique = new Set(ids);
    if (ids.length !== unique.size) {
      console.error('DUPLICATES in', LABEL_TO_FILENAME[label]);
      process.exit(1);
    }
  }
  console.log('No duplicates in any label file.');
}

main();
