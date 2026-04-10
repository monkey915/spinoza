#!/usr/bin/env node
/**
 * Capture a demo GIF of the spinoza replay viewer with robot arm animation.
 * Uses Puppeteer to control the WebUI and gif-encoder-2 to create the GIF.
 *
 * Usage: cd scripts && node capture_demo.js [--replay N] [--output ../docs/demo.gif]
 */

const puppeteer = require('puppeteer');
const GIFEncoder = require('gif-encoder-2');
const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');

const REPLAY_INDEX = parseInt(process.argv.find((_, i, a) => a[i-1] === '--replay') || '0');
const OUTPUT = process.argv.find((_, i, a) => a[i-1] === '--output') || '../docs/demo.gif';
const WIDTH = 800;
const HEIGHT = 500;
const FPS = 15;
const DURATION_SEC = 3.0;
const TOTAL_FRAMES = Math.floor(FPS * DURATION_SEC);

async function main() {
  const outPath = path.resolve(OUTPUT);
  console.log('Capturing replay #' + REPLAY_INDEX + ' -> ' + outPath);
  console.log(WIDTH + 'x' + HEIGHT + ', ' + TOTAL_FRAMES + ' frames @ ' + FPS + 'fps');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox',
           '--window-size=' + WIDTH + ',' + HEIGHT,
           '--enable-webgl', '--ignore-gpu-blocklist',
           '--use-gl=angle', '--use-angle=swiftshader',
           '--enable-unsafe-swiftshader'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT });

  // Collect console messages for debugging
  page.on('console', msg => {
    if (msg.type() === 'error' || msg.type() === 'warning') {
      console.log('  [browser ' + msg.type() + ']', msg.text());
    }
  });
  page.on('pageerror', err => console.log('  [page error]', err.message));

  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle2', timeout: 15000 });
  await page.waitForFunction(function() { return !!document.querySelector('canvas'); }, { timeout: 10000 });
  console.log('Page loaded, waiting for Three.js init...');
  await new Promise(function(r) { setTimeout(r, 3000); });

  // Click the load button
  await page.click('#replay-load');
  console.log('Clicked load button, waiting for replay data...');

  // Wait for the replay controls to appear (indicates successful load)
  await page.waitForFunction(function() {
    var el = document.getElementById('replay-controls');
    return el && el.style.display !== 'none';
  }, { timeout: 10000 });
  console.log('Replays loaded!');

  // Wait a bit for the first replay to render
  await new Promise(function(r) { setTimeout(r, 1000); });

  // Navigate to desired replay if not 0
  if (REPLAY_INDEX > 0) {
    for (var i = 0; i < REPLAY_INDEX; i++) {
      await page.click('#replay-next');
      await new Promise(function(r) { setTimeout(r, 300); });
    }
    await new Promise(function(r) { setTimeout(r, 500); });
  }

  // Set camera via exposed window globals
  await page.evaluate(function() {
    if (window.camera && window.s2t) {
      window.camera.position.copy(window.s2t(-0.8, 0.5, 2.0));
      window.controls.target.copy(window.s2t(0.76, 2.0, 0.76));
      window.controls.update();
    }
  });
  await new Promise(function(r) { setTimeout(r, 200); });

  // Press play
  await page.click('#btn-play');
  console.log('Playing replay...');
  await new Promise(function(r) { setTimeout(r, 100); });

  // Capture frames
  var encoder = new GIFEncoder(WIDTH, HEIGHT, 'neuquant', false, TOTAL_FRAMES);
  encoder.setDelay(Math.floor(1000 / FPS));
  encoder.setQuality(10);
  encoder.setRepeat(0);

  var gifStream = fs.createWriteStream(outPath);
  encoder.createReadStream().pipe(gifStream);
  encoder.start();

  console.log('Capturing ' + TOTAL_FRAMES + ' frames...');
  for (var i = 0; i < TOTAL_FRAMES; i++) {
    var screenshot = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT } });
    var png = PNG.sync.read(screenshot);
    encoder.addFrame(png.data);
    if (i % 10 === 0) process.stdout.write('  frame ' + i + '/' + TOTAL_FRAMES + '\r');
    await new Promise(function(r) { setTimeout(r, Math.floor(1000 / FPS)); });
  }

  encoder.finish();
  console.log('\nWaiting for GIF write...');

  await new Promise(function(resolve) { gifStream.on('finish', resolve); });
  var size = fs.statSync(outPath).size;
  console.log('GIF saved: ' + outPath + ' (' + Math.floor(size / 1024) + ' KB)');

  await browser.close();
}

main().catch(function(err) {
  console.error('Error:', err);
  process.exit(1);
});
