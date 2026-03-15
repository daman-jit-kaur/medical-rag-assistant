import urllib.request
import urllib.parse
import json
import time

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def search_pubmed(query, max_results=10):
    """Search PubMed and return list of paper IDs."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    })
    url = f"{PUBMED_SEARCH_URL}?{params}"
    
    try:
        response = urllib.request.urlopen(url, timeout=10)
        data = json.loads(response.read().decode())
        return data["esearchresult"]["idlist"]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def fetch_abstracts(paper_ids):
    """Fetch abstracts for a list of PubMed paper IDs."""
    if not paper_ids:
        return []
    
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(paper_ids),
        "retmode": "xml",
        "rettype": "abstract"
    })
    url = f"{PUBMED_FETCH_URL}?{params}"
    
    try:
        response = urllib.request.urlopen(url, timeout=15)
        xml_data = response.read().decode()
        return parse_abstracts_from_xml(xml_data)
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

def parse_abstracts_from_xml(xml_data):
    """Parse abstracts from PubMed XML response."""
    papers = []
    
    # Split by article
    articles = xml_data.split("<PubmedArticle>")
    
    for article in articles[1:]:  # Skip first empty split
        try:
            # Extract title
            title = ""
            if "<ArticleTitle>" in article:
                title_start = article.index("<ArticleTitle>") + len("<ArticleTitle>")
                title_end = article.index("</ArticleTitle>")
                title = article[title_start:title_end].strip()
                # Clean XML tags from title
                title = clean_xml_tags(title)
            
            # Extract abstract
            abstract = ""
            if "<AbstractText>" in article:
                abstract_start = article.index("<AbstractText>") + len("<AbstractText>")
                abstract_end = article.index("</AbstractText>")
                abstract = article[abstract_start:abstract_end].strip()
                abstract = clean_xml_tags(abstract)
            elif 'AbstractText Label=' in article:
                # Handle structured abstracts
                parts = article.split("<AbstractText")
                abstract_parts = []
                for part in parts[1:]:
                    if ">" in part and "</AbstractText>" in part:
                        text_start = part.index(">") + 1
                        text_end = part.index("</AbstractText>")
                        abstract_parts.append(clean_xml_tags(part[text_start:text_end]))
                abstract = " ".join(abstract_parts)
            
            # Extract PMID
            pmid = ""
            if "<PMID " in article or "<PMID>" in article:
                try:
                    if "<PMID>" in article:
                        pmid_start = article.index("<PMID>") + len("<PMID>")
                        pmid_end = article.index("</PMID>")
                    else:
                        pmid_start = article.index("<PMID ") 
                        pmid_start = article.index(">", pmid_start) + 1
                        pmid_end = article.index("</PMID>")
                    pmid = article[pmid_start:pmid_end].strip()
                except:
                    pass
            
            if title and abstract:
                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "source": f"PubMed:{pmid} - {title[:60]}..."
                })
        except Exception as e:
            continue
    
    return papers

def clean_xml_tags(text):
    """Remove XML tags from text."""
    import re
    clean = re.sub(r'<[^>]+>', '', text)
    clean = clean.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').replace('&quot;', '"')
    return clean.strip()

def get_papers_for_question(question, max_results=10):
    """Full pipeline: search PubMed and return paper abstracts."""
    print(f"  Searching PubMed for: '{question}'")
    
    # Search for paper IDs
    paper_ids = search_pubmed(question, max_results)
    
    if not paper_ids:
        return []
    
    print(f"  Found {len(paper_ids)} papers, fetching abstracts...")
    
    # Small delay to respect API rate limits
    time.sleep(0.5)
    
    # Fetch abstracts
    papers = fetch_abstracts(paper_ids)
    print(f"  ✅ Retrieved {len(papers)} abstracts")
    
    return papers