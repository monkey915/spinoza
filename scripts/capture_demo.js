#!/usr/bin/env node
/**
 * Capture a demo GIF of the spinoza replay viewer with robot arm animation.
 * Uses Puppeteer to control the WebUI and gif-encoder-2 to create the GIF.
 *
 * Usage: node scripts/capture_demo.js [--replay N] [--output docs/demo.gif]
 */

const puppeteer = require('puppeteer');
const GIFEncoder = require('gif-encoder-2');
const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');

const REPLAY_INDEX = parseInt(process.argv.find((_, i, a) => a[i-1] === '--replay') || '0');
const OUTPUT = process.argv.find((_, i, a) => a[i-1] === '--output') || 'docs/demo.gif';
const WIDTH = 800;
const HEIGHT = 500;
const FPS = 20;
const DURATION_SEC = 2.5;
const TOTAL_FRAMES = Math.floor(FPS * DURATION_SEC);

async function main() {
  console.log(`Capturing replay #${REPLAY_INDEX} → ${OUTPUT} (${WIDTH}x${HEIGHT}, ${TOTAL_FRAMES} frames @ ${FPS}fps)`);

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox',
           `--window-size=${WIDTH},${HEIGHT}`, '--use-gl=swiftshader'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT });

  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle2', timeout: 15000 });
  await page.waitForFunction(() => document.querySelector('canvas'), { timeout: 10000 });
  await new Promise(r => setTimeout(r, 2000));

  // Load replays, select replay, set camera, press play
  await page.evaluate(async (replayIdx) => {
    document.getElementById('replay-load').click();
    await new Promise(r => setTimeout(r, 2000));

    const slider = document.getElementById('replay-slider');
    if (slider) {
      slider.value = replayIdx;
      slider.dispatchEvent(new Event('input'));
    }
    await new Promise(r => setTimeout(r, 500));

    // Side-angle camera showing table + robot arm
    if (typeof camera !== 'undefined' && typeof controls !== 'undefined' && typeof s2t !== 'undefined') {
      camera.position.copy(s2t(-0.8, 0.5, 2.0));
      controls.target.copy(s2t(0.76, 2.0, 0.76));
      controls.update();
    }

    document.getElementById('btn-play').click();
  }, REPLAY_INDEX);

  await new Promise(r => setTimeout(r, 300));

  // Capture frames
  const encoder = new GIFEncoder(WIDTH, HEIGHT, 'neuquant', false, TOTAL_FRAMES);
  encoder.setDelay(Math.floor(1000 / FPS));
  encoder.setQuality(10);
  encoder.setRepeat(0);

  const gifStream = fs.createWriteStream(path.resolve(OUTPUT));
  encoder.createReadStream().pipe(gifStream);
  encoder.start();

  console.log('Capturing frames...');
  for (let i = 0; i < TOTAL_FRAMES; i++) {
    const screenshot = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT } });
    const png = PNG.sync.read(screenshot);
    encoder.addFrame(png.data);

    if (i % 10 === 0) process.stdout.write(`  frame ${i}/${TOTAL_FRAMES}\r`);
    await new Promise(r => setTimeout(r, Math.floor(1000 / FPS)));
  }

  encoder.finish();
  console.log(`\nWaiting for GIF write...`);

  await new Promise((resolve) => gifStream.on('finish', resolve));
  const size = fs.statSync(path.resolve(OUTPUT)).size;
  console.log(`GIF saved: ${OUTPUT} (${(size / 1024).toFixed(0)} KB)`);

  await browser.close();
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
