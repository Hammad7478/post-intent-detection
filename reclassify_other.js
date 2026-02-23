const fs = require('fs');
const path = require('path');

const annotatedDir = path.join(__dirname, 'data', 'annotated');

const LABEL_FILES = {
  ADVICE_SEEKING: 'advice_seeking_modelA.json',
  PERSONAL_EXPERIENCE: 'personal_experience_modelA.json',
  OPINION: 'opinion_modelA.json',
  OTHER: 'other_modelA.json',
};

/**
 * Reclassify text that was previously labeled OTHER.
 * Uses broader patterns so more posts get ADVICE_SEEKING, PERSONAL_EXPERIENCE, or OPINION.
 */
function reclassify(text) {
  const t = (text || '').toLowerCase().trim();
  const wordCount = t.split(/\s+/).filter(Boolean).length;

  // Step 1: ADVICE_SEEKING - request for help, input, perspectives, recommendations
  const advicePatterns = [
    /\b(wondering if|wondering whether)\s+(this is|that is|it's)/,
    /\b(perspective|perspectives)\s+(would\s+)?(help|appreciated)/,
    /\bany\s+(help|leads|direction)\s+(would\s+)?(be\s+)?(really\s+)?(helpful|appreciated)/,
    /\bcurious\s+(what|if|how|whether)/,
    /\b(tell me|someone tell me)\s+what\s+(i'm|i am)\s+doing wrong/,
    /\bany\s+insights?\s+(would\s+be\s+)?(really\s+)?helpful/,
    /\b(get|getting)\s+some\s+perspectives?/,
    /\bnot\s+sure\s+which\s+(to|one)/,
    /\b(which\s+to\s+)?lean\s+towards?/,
    /\b(tough\s+time|hard\s+time)\s+deciding/,
    /\bhaving\s+a\s+tough\s+time\s+deciding/,
    /\b(i'd|i would)\s+appreciate\s+(insights?|any\s+advice|your\s+thoughts|input)/,
    /\btrying\s+to\s+decide\s+whether/,
    /\bmy\s+only\s+question\s+is/,
    /\bwhat(\s+is|\s's)\s+the\s+best\s+(method|way|strategy|approach|option|use)/,
    /\boutside\s+perspectives?/,
    /\ball\s+advice\s+(is\s+)?appreciated/,
    /\bthank\s+you\s+in\s+advance\s+for\s+any\s+(replies|help|advice)/,
    /\bthanks?\s+for\s+any(one's|one\s+else's)?\s+(opinion|advice|thoughts|input)/,
    /\bappreciate\s+(the\s+community'?s?\s+)?insights?/,
    /\blooking\s+for\s+(general\s+)?advice/,
    /\b(just\s+)?(looking\s+for|open\s+to)\s+(all\s+)?(suggestions?|advice|recommendations?)/,
    /\b(figure\s+out|trying\s+to\s+figure\s+out)\s+what\s+(is\s+)?(feasible|best|realistic)/,
    /\bi(\s+am|\s+'m)\s+deciding\s+on/,
    /\bjust\s+wanted\s+to\s+know\s+if/,
    /\b(anything|any\s+direction)\s+you\s+(could\s+)?(kind\s+)?(people\s+)?could\s+point\s+me\s+in/,
    /\bi'?\s*m\s+debating\s+whether/,
    /\bhave\s+no\s+idea\s+what\s+(an?\s+)?(okay|ok)\s+amount\s+is/,
    /\bi'?\s*m\s+not\s+sure\s+if\s+it\s+would\s+be\s+wise/,
    /\bcan\s+anyone\s+recommend/,
    /\bcurious\s+if\s+anyone\s+else\s+has\s+(done|experience)/,
    /\b(am\s+i\s+better\s+to|which\s+is\s+the\s+best)\s+(use\s+of\s+money|to\s+overpay)?/,
    /\bsome\s+help\s+would\s+come\s+(a\s+)?long\s+way/,
    /\bhear\s+how\s+others\s+(are\s+)?(approaching|managing|doing)/,
    /\blooking\s+for\s+any\s+and\s+all\s+feedback/,
    /\bwhat(\s+is|\'s)\s+the\s+best\s+strategy\s+to/,
    /\bthanks?\s+for\s+the\s+advice/,
    /\bim?\s+open\s+to\s+all\s+suggestions?/,
    /\banything\s+is\s+appreciated/,
    /\bwould\s+love\s+your\s+(thoughts?|advice|input)/,
    /\bthoughts?\s+and\s+(suggestions?|advice)\s+please/,
    /\bappreciate\s+the\s+advice/,
    /\b(open\s+for|looking\s+for)\s+all\s+advice/,
    /\byour\s+insight(s)?\s+will\s+be\s+greatly\s+appreciated/,
    /\b(guide\s+me|please\s+guide\s+me)/,
    /\bhoping\s+someone\s+(out\s+there\s+)?has\s+(career\s+)?(ideas|suggestions?)/,
    /\bplease\s+let\s+me\s+know\s+if/,
    /\bwant\s+to\s+know\s+if\s+there'?s\s+any/,
    /\b(i'?\s*m\s+)?(looking\s+for|seeking)\s+(your\s+)?advice\s+(on|in|please)?/,
    /\btruly\s+anything\s+(direction\s+)?you\s+could\s+point\s+me/,
    /\breally\s+looking\s+for\s+any\s+applicable\s+advice/,
    /\bwhat\s+i\s+am\s+debating\s+is/,
    /\bwhich\s+is\s+the\s+best\s+use\s+of\s+money/,
    /\bdon'?t\s+know\s+how\s+realistic\s+/,
    /\bjust\s+trying\s+to\s+figure\s+out\s+what/,
    /\b(i'?\s*m\s+)?(very\s+)?confused\s+(between|about)/,
    /\b(thoughts?|advice|suggestions?)\s+please\s*[.!]?\s*$/,
    /\bthanks?\s+in\s+advance/,
    /\bthank\s+you\s+all\s+(so\s+)?much\s*[.!]?\s*$/,
    /\bany(thing|one)\s+(is\s+)?(appreciated|helpful)\s*[.!]?\s*$/,
    /\bwould\s+appreciate\s+(any|some)\s+(thoughts?|input|advice)/,
    /\bnot\s+sure\s+(what|how|whether|if)\s+/,
    /\b(i'?\s*m\s+)?torn\b/,
    /\bim\s+torn\b/,
    /\bany\s+help\s+or\s+leads\s+would\s+be\s+really\s+helpful\b/,
    /\bany\s+insights?\s+from\s+.*would\s+be\s+really\s+helpful\b/,
    /\bin\s+a\s+predicament\b/,
    /\bnot\s+sure\s+which\s+(stack|language|option)\s+/,
    /\bwhich\s+stack\s+or\s+language\s+i\s+should\s+choose\b/,
    /\bi'?\s*m\s+hesitant\s+because\b/,
    /\bi\s+am\s+leaning\s+towards\b/,
    /\bi'?\s*m\s+debating\s+whether\b/,
    /\bdebating\s+whether\s+(i\s+should|to)\b/,
    /\bi\s+kinda\s+wanna\s+know\s+what\b/,
    /\bwhat\s+would\s+be\s+a\s+complete\s+waste\s+of\s+money\b/,
    /\b(how\s+)?(much|what).*\s+to\s+spend\b.*\b(first\s+apartment|responsible)\b/,
    /\bscared\s+to\s+tell\s+(my\s+)?(boss|manager)\b/,
    /\bideally\s+i'?d\s+like\s+(a\s+)?role\s+that\s+paid\s+more\b/,
    /\bremote\s+job\s+would\s+be\s+(better|a\s+better)\b/,
    /\b(i'?\s*m\s+)?wondering\s+if\s+pursuing\b/,
    /\bwhat\s+i'?\s*m\s+doing\s+wrong\b/,
    /\bsomeone\s+tell\s+me\s+what\b/,
    /\bin\s+a\s+situation\s+where\b/,
    /\bscared\s+to\s+tell\s+.*\s+boss\b/,
    /\bappreciate\s+the\s+community'?\s*s?\s*insights?\b/,
    /\bfirst\s+apartment\b.*\b(want\s+to\s+be\s+)?responsible\b/,
    /\bdon'?t\s+know\s+how\s+much\s+longer\s+i\s+can\b/,
    /\bthought\s+about\s+majoring\s+in\s+cs\b/,
  ];
  if (advicePatterns.some(p => p.test(t))) return 'ADVICE_SEEKING';

  // Questions that are clearly asking for input (end with ? and ask for advice/thoughts)
  if (/\?\s*$/.test(t) && (/\b(common|normal|feasible|wise|better|best|okay|realistic)\s*\?\s*$/.test(t) || /\b(what|how|which|should|would|can)\b.*\?\s*$/.test(t) && wordCount > 20)) return 'ADVICE_SEEKING';

  // Step 2: PERSONAL_EXPERIENCE - first-person narrative, no clear request
  const narrativePatterns = [
    /\bwe\s+ended\s+up\s+keeping\s+things\s+remote\b/,
    /\bit\s+is\s+not\s+the\s+path\s+i\s+thought\s+would\s+take\b/,
    /\bthe\s+more\s+complicated\s+relocation\s+gets\b/,
  ];
  const hasClearAsk = /\b(advice|suggestions?|recommendations?|what\s+should|how\s+(do|can)\s+i|should\s+i\s+|thoughts?\s*\?|please\s+(tell|advise|help)|would\s+love\s+your|appreciate\s+(any|your)|looking\s+for\s+(advice|suggestions?)|any\s+help\b)/.test(t);
  if (!hasClearAsk && narrativePatterns.some(p => p.test(t)) && wordCount > 40) return 'PERSONAL_EXPERIENCE';

  // Step 3: OPINION - sharing link/news with commentary, predictions, rants
  const opinionPatterns = [
    /^https?:\/\//,
    /\b(shame\s+because|feel\s+really\s+bad\s+for)\b/,
    /\bwill\s+make\s+a\s+killing\b/,
    /\bit'?s\s+almost\s+as\s+if\b/,
    /\bi\s+would\s+say\s+investors\s+are\s+being\s+duped\b/,
    /\bthe\s+golden\s+age\s+.*\s+is\s+over\b/,
    /\bthis\s+post\s+isn'?t\s+to\s+straight\s+up\s+tell\s+you\b/,
    /\bgatekeepers?\s+are\s+annoying\b/,
    /\bthe\s+doom\s+and\s+gloom\s+in\s+this\s+field\b/,
    /\bgood\s+luck\s+out\s+there\b/,
    /\b(i\s+)?(think|feel|believe)\s+(this\s+is|that)\s+common\b/,
    /\bthe\s+more\s+i\s+see\s+.*\s+feel\s+like\s+to\s+a\s+whole\s+different\s+level\b/,
    /\bplease\s+don'?t\s+talk\s+like\b/,
    /\bthe\s+reality\s+is\s+.*\s+you\s+need\s+to\s+accept\b/,
    /\bai\s*,?\s*market\s+is\s+cooked\b/,
    /\bceos?\s+.*\s+wet\s+dreams?\s+to\s+replace\b/,
    /\bi'?\s*m\s+failing\s+to\s+see\s+where\b/,
    /\bi\s+looked\s+into\s+the\s+trades\s+and\s+it\s+seems\b/,
  ];
  if (opinionPatterns.some(p => p.test(t)) && wordCount > 25) return 'OPINION';

  return 'OTHER';
}

function main() {
  const byLabel = {
    ADVICE_SEEKING: [],
    PERSONAL_EXPERIENCE: [],
    OPINION: [],
    OTHER: [],
  };

  for (const [label, filename] of Object.entries(LABEL_FILES)) {
    const filePath = path.join(annotatedDir, filename);
    if (!fs.existsSync(filePath)) {
      console.error('Missing:', filename);
      process.exit(1);
    }
    byLabel[label] = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  }

  const otherPosts = byLabel.OTHER;
  const toRemove = [];
  const stats = { ADVICE_SEEKING: 0, PERSONAL_EXPERIENCE: 0, OPINION: 0, OTHER: 0 };

  for (let i = 0; i < otherPosts.length; i++) {
    const post = otherPosts[i];
    const newLabel = reclassify(post.text);
    if (newLabel !== 'OTHER') {
      post.label = newLabel;
      byLabel[newLabel].push(post);
      toRemove.push(i);
      stats[newLabel]++;
    } else {
      stats.OTHER++;
    }
  }

  byLabel.OTHER = otherPosts.filter((_, i) => !toRemove.includes(i));

  for (const [label, filename] of Object.entries(LABEL_FILES)) {
    const filePath = path.join(annotatedDir, filename);
    fs.writeFileSync(filePath, JSON.stringify(byLabel[label], null, 2), 'utf8');
    console.log(filename, byLabel[label].length, 'posts');
  }

  const moved = toRemove.length;
  console.log('\nReclassified', moved, 'posts from OTHER to another category.');
  console.log('New assignments:', stats);
  console.log('Remaining in OTHER:', stats.OTHER);
}

main();
