import feedparser
import json
import os
import urllib.parse

def fetch_arxiv_papers(query="llm benchmark", max_results=5):
    encoded_query = urllib.parse.quote(query)
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
    
    feed = feedparser.parse(url)
    
    papers = []
    
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]  # 🔥 important

        paper = {
            "arxiv_id": arxiv_id,
            "title": entry.title,
            "authors": [a.name for a in entry.authors],
            "abstract": entry.summary,
            "link": entry.link,
            "pdf_link": f"https://arxiv.org/pdf/{arxiv_id}.pdf", 
            "published": entry.published,
        }

        papers.append(paper)
    
    return papers


def save_papers(papers, filepath="data/papers.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=4)