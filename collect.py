import yfinance as yf
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
import requests
import time

class StockDataCollector:
    def __init__(self, symbol, mongodb_uri):
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
        self.client = MongoClient(mongodb_uri)
        self.db = self.client['stock_data']
        
    def collect_basic_info(self):
        """收集基本股票数据(OHLCV)和当前财务指标"""
        collection = self.db['basic_info']
        
        # 获取历史OHLCV数据
        hist_data = self.stock.history(period="max")
        
        # 获取当前财务指标
        info = self.stock.info
        
        basic_data = {
            'symbol': self.symbol,
            'last_updated': datetime.now(),
            'price_data': hist_data.to_dict('records'),
            'current_metrics': {
                'eps': info.get('trailingEPS'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargin')
            }
        }
        
        collection.update_one(
            {'symbol': self.symbol},
            {'$set': basic_data},
            upsert=True
        )
        
    def collect_historical_financials(self):
        """收集历史财务数据，包括季度和年度数据"""
        collection = self.db['historical_financials']
        
        try:
            # 获取财务报表数据
            # 资产负债表
            balance_sheet_y = self.stock.balance_sheet  # 年度数据
            balance_sheet_q = self.stock.quarterly_balance_sheet  # 季度数据
            
            # 利润表
            income_stmt_y = self.stock.income_stmt  # 年度数据
            income_stmt_q = self.stock.quarterly_income_stmt  # 季度数据
            
            # 现金流量表
            cash_flow_y = self.stock.cash_flow  # 年度数据
            cash_flow_q = self.stock.quarterly_cash_flow  # 季度数据
            
            # 整理财务数据
            financials_data = {
                'symbol': self.symbol,
                'last_updated': datetime.now(),
                'annual_data': {
                    'balance_sheet': balance_sheet_y.to_dict() if balance_sheet_y is not None else {},
                    'income_statement': income_stmt_y.to_dict() if income_stmt_y is not None else {},
                    'cash_flow': cash_flow_y.to_dict() if cash_flow_y is not None else {}
                },
                'quarterly_data': {
                    'balance_sheet': balance_sheet_q.to_dict() if balance_sheet_q is not None else {},
                    'income_statement': income_stmt_q.to_dict() if income_stmt_q is not None else {},
                    'cash_flow': cash_flow_q.to_dict() if cash_flow_q is not None else {}
                }
            }
            
            # 计算历史财务比率
            self._calculate_historical_ratios(financials_data)
            
            # 更新数据库
            collection.update_one(
                {'symbol': self.symbol},
                {'$set': financials_data},
                upsert=True
            )
            
            print(f"成功收集 {self.symbol} 的历史财务数据")
            return True
            
        except Exception as e:
            print(f"收集历史财务数据时发生错误: {str(e)}")
            return False
    
    def _calculate_historical_ratios(self, financials_data):
        """计算历史财务比率"""
        try:
            # 获取历史股价数据用于计算市值相关指标
            hist_prices = self.stock.history(period="max")
            
            # 准备季度数据
            income_q = pd.DataFrame(financials_data['quarterly_data']['income_statement'])
            balance_q = pd.DataFrame(financials_data['quarterly_data']['balance_sheet'])
            
            # 计算季度关键指标
            quarterly_ratios = {}
            for date in income_q.columns:
                try:
                    # 获取该季度末的股价
                    price_date = pd.to_datetime(date)
                    closest_price = hist_prices.asof(price_date)['Close'] if not hist_prices.empty else None
                    
                    # 基本每股收益
                    net_income = income_q[date].get('Net Income', None)
                    shares = balance_q[date].get('Share Issued', None)
                    eps = net_income / shares if net_income is not None and shares is not None else None
                    
                    # 市盈率 (P/E)
                    pe_ratio = closest_price / eps if eps and closest_price else None
                    
                    # 市净率 (P/B)
                    total_equity = balance_q[date].get('Total Stockholder Equity', None)
                    book_value_per_share = total_equity / shares if total_equity is not None and shares is not None else None
                    pb_ratio = closest_price / book_value_per_share if book_value_per_share and closest_price else None
                    
                    # ROA (Return on Assets)
                    total_assets = balance_q[date].get('Total Assets', None)
                    roa = (net_income / total_assets) if total_assets and net_income else None
                    
                    # ROE (Return on Equity)
                    roe = (net_income / total_equity) if total_equity and net_income else None
                    
                    # 毛利率 (Gross Margin)
                    revenue = income_q[date].get('Total Revenue', None)
                    gross_profit = income_q[date].get('Gross Profit', None)
                    gross_margin = (gross_profit / revenue) if revenue and gross_profit else None
                    
                    quarterly_ratios[str(date)] = {
                        'eps': eps,
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'roa': roa,
                        'roe': roe,
                        'gross_margin': gross_margin,
                        'close_price': closest_price
                    }
                    
                except Exception as e:
                    print(f"计算 {date} 的财务比率时发生错误: {str(e)}")
                    continue
            
            # 将计算的比率添加到财务数据中
            financials_data['quarterly_ratios'] = quarterly_ratios
            
        except Exception as e:
            print(f"计算历史财务比率时发生错误: {str(e)}")
    
    def collect_options_data(self):
        """收集期权数据"""
        collection = self.db['options_data']
        
        # 获取所有可用的期权到期日
        expiration_dates = self.stock.options
        
        options_data = []
        for date in expiration_dates:
            opt = self.stock.option_chain(date)
            
            # 合并看涨和看跌期权数据
            calls = opt.calls.to_dict('records')
            puts = opt.puts.to_dict('records')
            
            options_data.append({
                'expiration_date': date,
                'calls': calls,
                'puts': puts
            })
            
        collection.update_one(
            {'symbol': self.symbol},
            {
                '$set': {
                    'last_updated': datetime.now(),
                    'options_chain': options_data
                }
            },
            upsert=True
        )
        
    def collect_analyst_ratings(self):
        """
        收集分析师评级数据，并保留历史记录
        每次收集的数据会作为新的记录追加到数据库中
        """
        collection = self.db['analyst_ratings']
    
        # 获取分析师推荐
        recommendations = self.stock.recommendations
        if recommendations is not None:
            # 添加收集时间戳
            current_time = datetime.now()
        
            # 转换recommendations为列表格式并添加采集时间
            recommendations_list = recommendations.to_dict('records')
            for record in recommendations_list:
                record['collected_at'] = current_time
        
            # 更新数据库，使用$push将新数据追加到历史记录中
            collection.update_one(
                {'symbol': self.symbol},
                {
                    '$set': {
                        'last_updated': current_time,
                    },
                    '$push': {
                        'historical_recommendations': {
                            '$each': recommendations_list
                        }
                    }
                },
                upsert=True
            )
        
            print(f"成功收集 {self.symbol} 的分析师评级数据，共 {len(recommendations_list)} 条记录")
            return True
        else:
            print(f"未能获取到 {self.symbol} 的分析师评级数据")
            return False
            
    def collect_industry_data(self):
        """收集行业和市场指数数据"""
        collection = self.db['industry_data']
        
        # 获取标普500指数数据作为市场指标
        sp500 = yf.Ticker('^GSPC')
        # 获取科技行业ETF数据作为行业指标
        tech_etf = yf.Ticker('XLK')
        
        market_data = {
            'symbol': self.symbol,
            'last_updated': datetime.now(),
            'market_index': sp500.history(period="1y").to_dict('records'),
            'industry_index': tech_etf.history(period="1y").to_dict('records')
        }
        
        collection.update_one(
            {'symbol': self.symbol},
            {'$set': market_data},
            upsert=True
        )
        
    def collect_peer_data(self):
        """收集同行业公司数据"""
        collection = self.db['peer_data']
        
        # 苹果的主要竞争对手
        peers = ['MSFT', 'GOOGL', 'AMZN', 'META']
        
        peer_data = {
            'symbol': self.symbol,
            'last_updated': datetime.now(),
            'peer_prices': {}
        }
        
        for peer in peers:
            peer_stock = yf.Ticker(peer)
            peer_data['peer_prices'][peer] = peer_stock.history(period="1y").to_dict('records')
            
        collection.update_one(
            {'symbol': self.symbol},
            {'$set': peer_data},
            upsert=True
        )
    
    def collect_all_data(self):
        """收集所有数据"""
        print(f"开始收集 {self.symbol} 的所有数据...")
        
        # 收集基本信息
        try:
            self.collect_basic_info()
            print("基本信息收集完成")
        except Exception as e:
            print(f"收集基本信息时出错: {str(e)}")
        
        # 收集历史财务数据
        try:
            self.collect_historical_financials()
            print("历史财务数据收集完成")
        except Exception as e:
            print(f"收集历史财务数据时出错: {str(e)}")
        
        # 收集期权数据
        try:
            self.collect_options_data()
            print("期权数据收集完成")
        except Exception as e:
            print(f"收集期权数据时出错: {str(e)}")
        
        # 收集分析师评级
        try:
            self.collect_analyst_ratings()
            print("分析师评级数据收集完成")
        except Exception as e:
            print(f"收集分析师评级时出错: {str(e)}")
        
        # 收集行业数据
        try:
            self.collect_industry_data()
            print("行业数据收集完成")
        except Exception as e:
            print(f"收集行业数据时出错: {str(e)}")
        
        # 收集同行数据
        try:
            self.collect_peer_data()
            print("同行数据收集完成")
        except Exception as e:
            print(f"收集同行数据时出错: {str(e)}")
        
        print(f"{self.symbol} 的所有数据收集完成")

# 使用示例
if __name__ == "__main__":
    # 替换为你的MongoDB连接URI
    MONGODB_URI = "mongodb://localhost:27017/"
    collector = StockDataCollector("AAPL", MONGODB_URI)
    
    # 收集所有数据
    collector.collect_all_data()