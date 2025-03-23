import asyncio
import os
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright
import aiohttp


async def fetch(drug_list):
    with open(os.path.join(data_dir, 'mapped_drug.csv'), 'w') as f:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            for drug in drug_list:
                if drug != 'arava':
                    continue
                print(drug)
                # exit()
                fetching_url = f'https://go.drugbank.com/unearth/q?searcher=drugs&query={quote_plus(drug)}'
                page = await browser.new_page()
                response = await page.goto(fetching_url)
                # await page.wait_for_timeout(3000)
                # await page.get_by_text('Leflunomide').click()
                # await page.wait_for_url('**/drugs/*')
                text = await response.text()
                if response is None:
                    print('No response')
                # async with session.get(fetching_url) as response:


if __name__ == "__main__":
    data_dir = '../data/task1/drugbank_mine_names'
    with open(os.path.join(data_dir, 'source.csv')) as f:
        drug_names = [l.strip() for l in f.readlines()]

    asyncio.run(fetch(drug_names))