import pandas as pd

wl = pd.read_csv('out/watchlist.csv', encoding='utf-8-sig')
verified = wl[wl['next_rise'].notna()]

print(f'=== 戦略B 累積成績 ===')
print(f'総件数: {len(verified)}件')

hit5  = (verified['next_rise'] >= 5).sum()
hit2  = (verified['next_rise'] >= 2).sum()
hit0  = (verified['next_rise'] >= 0).sum()
loss  = (verified['next_rise'] < 0).sum()
loss5 = (verified['next_rise'] < -5).sum()

print(f'+5%以上:  {hit5}件 ({hit5/len(verified)*100:.1f}%)')
print(f'+2%以上:  {hit2}件 ({hit2/len(verified)*100:.1f}%)')
print(f'プラス:   {hit0}件 ({hit0/len(verified)*100:.1f}%)')
print(f'マイナス: {loss}件 ({loss/len(verified)*100:.1f}%)')
print(f'-5%以下:  {loss5}件 ({loss5/len(verified)*100:.1f}%)')
print(f'平均:     {verified["next_rise"].mean():+.2f}%')
print(f'中央値:   {verified["next_rise"].median():+.2f}%')

print(f'\n=== RBスコア別 ===')
for score in sorted(verified['rebound_score'].dropna().unique()):
    s = verified[verified['rebound_score'] == score]
    h = (s['next_rise'] >= 5).sum()
    avg = s['next_rise'].mean()
    print(f'  {int(score)}点: {len(s):>3}件  +5%達成:{h}件({h/len(s)*100:.1f}%)  平均:{avg:+.2f}%')