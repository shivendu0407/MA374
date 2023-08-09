import requests
headers = {
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding':'gzip, deflate, br',
    'Connection':'keep-alive',
    'Cookie':'AKA_A2=A; ak_bmsc=2C30C88FD1C6BEED087CCD02E7643772~000000000000000000000000000000~YAAQPvASAr+cI2F6AQAAk6+tdQxncocfEFby+qeRnNu3MgRblj1MWVtVy+W1Stx/CNaRaf9PhfVoT568zV8qztByVrxV+WfdrCN2nXU0nToPdEoaZFeZ7irUu8aSUXcln/sou0taKkr1gjmS3f6faZs+Rv8LA32eUAtlTD+GgYL0OKTJ44qVVinDxeeaVOiLxzQaiv0YjRCLcovFhO7jVBCJhNeXzgOeUYCLjkOg+2DEnRaF1Cd85f83pkjjieOFpjvywz20ImVWy1fr+S2nEDqmcgKZdhjHPfJ76+Z3bvVB/Kyv2dH7J8BMjlVf7kxyGbmot54yxchJNEMs0A/QTkeow2Xa54IcGZo/RUxGRu90SFu6VpfcxLaVOdN9EbvhcNs//OPA1jhDm9Nf4A==; bm_sv=BB4B29FC4D88791AABD65B43FACB0AF7~ObLG1UzBN4vOInl5m0vWqjOpZUXtLDHJDxr92uXdHHp5bjKjrEMMJcJRzS5VY5lkIs3N7JH+gZtoTnYIWKFqPZFhFC8Oo+sjmZLrin4taKkPfpvp7RdbqySQh6BLQwbWg3UgQJUQN29H0q9MJN6FuaW2b2i13zn5CmZUSDSpJVo=',
    'Host':'www.nseindia.com',
    'Sec-Fetch-Dest':'document',
    'Sec-Fetch-Mode':'navigate',
    'Sec-Fetch-Site':'none',
    'Sec-Fetch-User':'?1',
    'TE':'trailers',
    'Upgrade-Insecure-Requests':'1',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
}

r = requests.get('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY', headers=headers)
print(r.json())