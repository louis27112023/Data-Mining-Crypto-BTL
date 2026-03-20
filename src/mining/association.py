import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_association_mining(df, config):
    """
    Khai phá dữ liệu: Luật Kết hợp (Association Rules)
    Áp dụng thuật toán Apriori trên giỏ hàng (Basket) theo từng ngày.
    Trong bối cảnh Crypto:
    - Item: Cặp "TênCoin_TrạngThái(Up/Down)" ví dụ "Bitcoin_Up"
    - Transaction ID (Basket): Ngày giao dịch (Date)
    """
    mining_cfg = config.get('mining', {}).get('association', {})
    min_support = mining_cfg.get('min_support', 0.1)
    min_confidence = mining_cfg.get('min_confidence', 0.5)
    
    print(f"Bắt đầu Association Rule Mining (min_support={min_support}, min_conf={min_confidence})...")
    
    # 1. Tạo giỏ hàng cơ sở
    df['Item'] = df['Symbol'] + "_" + df['Price_Trend']
    
    # Do 1 ngày có thể có nhiều đồng coin, ta lấy Vol_Regime đại diện (mode/first) của ngày đó
    daily_regime = df.groupby('Date')['Vol_Regime'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Normal_Vol')
    
    basket = (df.groupby(['Date', 'Item'])['Item']
              .count().unstack().reset_index().fillna(0)
              .set_index('Date'))
              
    def encode_units(x):
        return True if x >= 1 else False
        
    basket_sets = basket.map(encode_units)
    basket_sets = basket_sets.dropna(how='all')
    
    if basket_sets.empty:
        print("Dữ liệu giỏ hàng trống.")
        return None, None
        
    # Hàm con chạy luật
    def get_rules(b_sets, regime_name):
        frequent_itemsets = apriori(b_sets, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty: return pd.DataFrame()
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        if not rules.empty:
            rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
            rules['Regime'] = regime_name
        return rules

    print(f"Tổng số giao dịch (ngày): {len(basket_sets)}")
    
    # Chạy trên toàn bộ
    rules_all = get_rules(basket_sets, "All")
    
    # Tách giỏ hàng theo Regime
    basket_high = basket_sets[basket_sets.index.map(daily_regime) == 'High_Vol']
    basket_low = basket_sets[basket_sets.index.map(daily_regime).isin(['Low_Vol', 'Normal_Vol'])] # Gộp low và normal cho đỡ nhiễu
    
    rules_high = get_rules(basket_high, "High_Vol") if not basket_high.empty else pd.DataFrame()
    rules_low = get_rules(basket_low, "Low_Vol_Normal") if not basket_low.empty else pd.DataFrame()
    
    # Gộp tất cả luật
    all_rules_df = pd.concat([rules_all, rules_high, rules_low], ignore_index=True) if not rules_all.empty else pd.DataFrame()
    
    print(f"Tổng hợp được {len(all_rules_df)} luật kết hợp từ các regime khác nhau.")
    # Return fake frequent itemsets empty so pipeline doesn't break, and the full rules df
    return pd.DataFrame(), all_rules_df

if __name__ == "__main__":
    print("Test association mining module")
