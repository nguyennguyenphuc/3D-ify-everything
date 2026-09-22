import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1200}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:8000');
const projectId=process.argv[2] || 'bb4c5973eeea47ccbf389d5b95ef657f';
const completedId=process.argv[3] || '62bcc7ec64ad4d9fab59478696463bad';
await page.getByLabel('Project',{exact:true}).selectOption(projectId);
await page.getByRole('button',{name:'Splat 3D',exact:true}).waitFor();
await page.getByRole('status').filter({hasText:'Gaussian Splatting'}).waitFor({timeout:60000});
await page.waitForTimeout(3000);
await fs.mkdir('../data/viewer-verification',{recursive:true});
await page.locator('.result-viewer').screenshot({path:'../data/viewer-verification/splat.png'});
await page.getByRole('button',{name:'Point cloud',exact:true}).click();
await page.getByRole('status').filter({hasText:'hình học thưa'}).waitFor();
await page.waitForTimeout(500);
await page.locator('.result-viewer').screenshot({path:'../data/viewer-verification/points.png'});
await page.getByLabel('Hiện camera').check();
await page.getByLabel('Góc nhìn').selectOption('all');
await page.getByRole('status').filter({hasText:'hình học thưa'}).waitFor();
await page.waitForTimeout(500);
await page.locator('.result-viewer').screenshot({path:'../data/viewer-verification/cameras.png'});
await page.getByRole('button',{name:'Depth',exact:true}).click();
await page.locator('.depth-pair img').evaluateAll(async images=>{await Promise.all(images.map(i=>i.decode()))});
await page.locator('.result-viewer').screenshot({path:'../data/viewer-verification/depth.png'});
await page.reload();
await page.getByRole('status').filter({hasText:'Gaussian Splatting'}).waitFor({timeout:60000});
// The default fixture also exercises completion without rerunning GPU. Passing
// a project argument is a visual inspection mode for a supplied run.
if (!process.argv[2]) {
  const id=completedId;
  const completed=await (await page.request.get(`http://127.0.0.1:8000/api/jobs/${id}`)).json();
  let pending=true, connections=0;
  await page.route('**/api/projects/*/jobs',async route=>{
    const response=await route.fetch();const jobs=await response.json();
    await route.fulfill({json:jobs.map(j=>pending&&j.id===id?{...j,status:'running'}:j)});
  });
  await page.route(`**/api/jobs/${id}/events`,async route=>{
    connections++;pending=false;
    await route.fulfill({contentType:'text/event-stream',body:`event: done\ndata: ${JSON.stringify(completed)}\n\n`});
  });
  await page.reload();
  await page.getByRole('status').filter({hasText:'Gaussian Splatting'}).waitFor({timeout:60000});
  await page.waitForTimeout(1000);
  if(connections!==1)throw Error(`Expected one SSE completion connection, received ${connections}`);
}
console.log(JSON.stringify({errors,modes:'splat, points, cameras, depth'},null,2));
await browser.close();
if(errors.length)process.exitCode=1;
