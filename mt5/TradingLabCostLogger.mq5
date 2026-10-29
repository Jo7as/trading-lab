//+------------------------------------------------------------------+
//|                                        TradingLabCostLogger.mq5    |
//|  trading-lab: measures the REAL costs of a broker / prop account    |
//|  (spreads, swaps, commission, slippage) for every market of the    |
//|  S-007 trend portfolio. Read by trading_lab/broker_costs.py.        |
//|                                                                    |
//|  Passive by default: it only reads quotes and symbol properties.   |
//|  Optional slippage probe (InpProbe=true) sends minimal market      |
//|  orders and closes them again - ONLY on a DEMO account (checked).  |
//|                                                                    |
//|  Files (MQL5/Files, or Common/Files with InpCommonFolder=true):     |
//|    tl_spreads_<login>.csv  every InpEverySeconds per symbol        |
//|    tl_symbols_<login>.csv  once per day: swaps, contract, modes    |
//|    tl_probes_<login>.csv   slippage probes (if enabled)            |
//|    tl_allsymbols_<login>.csv  all broker symbol names (once)       |
//+------------------------------------------------------------------+
#property copyright "trading-lab"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

input group "=== Symbols (lab=broker, comma separated) ==="
input string InpMap1 = "SP500=SP500m,US100=ND100m,US30=,GER40=GDAXIm,UK100=UK100,JP225=NI225";   // FundedNext names
input string InpMap2 = "EURUSD=EURUSD,GBPUSD=GBPUSD,USDJPY=USDJPY,AUDUSD=AUDUSD,NZDUSD=NZDUSD,USDCAD=USDCAD,USDCHF=USDCHF";
input string InpMap3 = "EURJPY=EURJPY,EURGBP=EURGBP,EURCHF=EURCHF,GBPJPY=GBPJPY,AUDJPY=AUDJPY,AUDNZD=AUDNZD,AUDCAD=AUDCAD";
input string InpMap4 = "XAUUSD=XAUUSD,XAGUSD=XAGUSD,BRENT=,WTI=,COPPER=,UST10Y=,BTCUSD=";        // not offered on FundedNext-Server3

input group "=== Logging ==="
input int    InpEverySeconds = 60;      // quote snapshot interval
input bool   InpCommonFolder = false;   // write to Common/Files instead of MQL5/Files

input group "=== Slippage probe (DEMO only) ==="
input bool   InpProbe        = false;   // send minimal market orders to measure fills
input int    InpProbeHourNY  = 15;      // New York hour of the daily probe (S-007 rebalances 15:00 NY)
input int    InpProbeHoldSec = 5;       // seconds until the probe is closed again
input ulong  InpMagic        = 20261029;

string   g_lab[];
string   g_sym[];
int      g_n = 0;
datetime g_lastSymbolsDay = 0;
int      g_lastProbeDay = -1;
CTrade   g_trade;

//+------------------------------------------------------------------+
int Flags() { return FILE_CSV | FILE_READ | FILE_WRITE | FILE_SHARE_READ | FILE_ANSI | (InpCommonFolder ? FILE_COMMON : 0); }

string FileName(const string kind)
{
   return StringFormat("tl_%s_%I64d.csv", kind, AccountInfoInteger(ACCOUNT_LOGIN));
}

int OpenAppend(const string kind, const string header)
{
   int h = FileOpen(FileName(kind), Flags(), ';');
   if(h == INVALID_HANDLE) { Print("cannot open ", FileName(kind), " err ", GetLastError()); return h; }
   if(FileSize(h) == 0) FileWriteString(h, header + "\r\n");
   FileSeek(h, 0, SEEK_END);
   return h;
}

void ParseMap(const string m)
{
   string pairs[];
   int k = StringSplit(m, ',', pairs);
   for(int i = 0; i < k; i++)
   {
      string kv[];
      if(StringSplit(pairs[i], '=', kv) != 2) continue;
      StringTrimLeft(kv[0]); StringTrimRight(kv[0]); StringTrimLeft(kv[1]); StringTrimRight(kv[1]);
      if(kv[1] == "") { Print("no broker symbol for ", kv[0], " - skipped"); continue; }
      if(!SymbolSelect(kv[1], true)) { Print("symbol not found at this broker: ", kv[1], " (", kv[0], ")"); continue; }
      ArrayResize(g_lab, g_n + 1); ArrayResize(g_sym, g_n + 1);
      g_lab[g_n] = kv[0]; g_sym[g_n] = kv[1]; g_n++;
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   ParseMap(InpMap1); ParseMap(InpMap2); ParseMap(InpMap3); ParseMap(InpMap4);
   PrintFormat("CostLogger: %d symbols mapped, files in %s", g_n, InpCommonFolder ? "Common/Files" : "MQL5/Files");
   if(InpProbe && AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_DEMO)
   {
      Print("InpProbe=true is only allowed on a DEMO account - EA stopped.");
      return INIT_PARAMETERS_INCORRECT;
   }
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   DumpAllSymbols();
   LogSymbols();
   EventSetTimer(MathMax(InpEverySeconds, 5));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

//+------------------------------------------------------------------+
void DumpAllSymbols()
{
   int h = FileOpen(FileName("allsymbols"), FILE_CSV | FILE_WRITE | FILE_ANSI | (InpCommonFolder ? FILE_COMMON : 0), ';');
   if(h == INVALID_HANDLE) return;
   FileWrite(h, "name", "description", "path");
   for(int i = 0; i < SymbolsTotal(false); i++)
   {
      string s = SymbolName(i, false);
      FileWrite(h, s, SymbolInfoString(s, SYMBOL_DESCRIPTION), SymbolInfoString(s, SYMBOL_PATH));
   }
   FileClose(h);
}

// swap of one lot for one night as a fraction p.a. of the notional (NaN if the mode is unknown)
double SwapPa(const string s, const double swap, const double price)
{
   long mode = SymbolInfoInteger(s, SYMBOL_SWAP_MODE);
   double tv = SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE), ts = SymbolInfoDouble(s, SYMBOL_TRADE_TICK_SIZE);
   double pt = SymbolInfoDouble(s, SYMBOL_POINT);
   if(price <= 0) return EMPTY_VALUE;
   if(mode == SYMBOL_SWAP_MODE_POINTS)            return swap * pt / price * 360.0;
   if(mode == SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT)  return (ts > 0 && tv > 0) ? swap / (price * tv / ts) * 360.0 : EMPTY_VALUE;
   if(mode == SYMBOL_SWAP_MODE_INTEREST_CURRENT || mode == SYMBOL_SWAP_MODE_INTEREST_OPEN) return swap / 100.0;
   return EMPTY_VALUE;                            // other modes: raw values are logged anyway
}

void LogSymbols()
{
   int h = OpenAppend("symbols", "utc;lab;symbol;bid;ask;digits;point;contract;tick_value;tick_size;swap_mode;"
                      "swap_long;swap_short;swap_long_pa;swap_short_pa;triple_day;calc_mode;volume_min;currency_profit;currency_margin");
   if(h == INVALID_HANDLE) return;
   for(int i = 0; i < g_n; i++)
   {
      string s = g_sym[i];
      double bid = SymbolInfoDouble(s, SYMBOL_BID), ask = SymbolInfoDouble(s, SYMBOL_ASK);
      double mid = (bid + ask) / 2.0;
      double sl = SymbolInfoDouble(s, SYMBOL_SWAP_LONG), ss = SymbolInfoDouble(s, SYMBOL_SWAP_SHORT);
      FileWrite(h, TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS), g_lab[i], s,
                DoubleToString(bid, 8), DoubleToString(ask, 8), SymbolInfoInteger(s, SYMBOL_DIGITS),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_POINT), 10), DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_CONTRACT_SIZE), 4),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE), 8), DoubleToString(SymbolInfoDouble(s, SYMBOL_TRADE_TICK_SIZE), 10),
                SymbolInfoInteger(s, SYMBOL_SWAP_MODE), DoubleToString(sl, 6), DoubleToString(ss, 6),
                DoubleToString(SwapPa(s, sl, mid), 6), DoubleToString(SwapPa(s, ss, mid), 6),
                SymbolInfoInteger(s, SYMBOL_SWAP_ROLLOVER3DAYS), SymbolInfoInteger(s, SYMBOL_TRADE_CALC_MODE),
                DoubleToString(SymbolInfoDouble(s, SYMBOL_VOLUME_MIN), 4),
                SymbolInfoString(s, SYMBOL_CURRENCY_PROFIT), SymbolInfoString(s, SYMBOL_CURRENCY_MARGIN));
   }
   FileClose(h);
   g_lastSymbolsDay = iTime(_Symbol, PERIOD_D1, 0);
}

void LogSpreads()
{
   int h = OpenAppend("spreads", "utc;lab;symbol;bid;ask;spread_bp");
   if(h == INVALID_HANDLE) return;
   for(int i = 0; i < g_n; i++)
   {
      MqlTick t;
      if(!SymbolInfoTick(g_sym[i], t) || t.bid <= 0 || t.ask <= 0) continue;
      double mid = (t.bid + t.ask) / 2.0;
      FileWrite(h, TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS), g_lab[i], g_sym[i],
                DoubleToString(t.bid, 8), DoubleToString(t.ask, 8), DoubleToString((t.ask - t.bid) / mid * 1e4, 4));
   }
   FileClose(h);
}

//+------------------------------------------------------------------+
// New York hour from GMT (US DST: 2nd Sunday of March .. 1st Sunday of November; the 2 a.m.
// switch hour itself is ignored - the probe runs at 15:00 NY)
int NyHour()
{
   MqlDateTime d; TimeToStruct(TimeGMT(), d);
   int dow1 = (d.day_of_week - (d.day - 1) % 7 + 7) % 7;     // weekday of the 1st of the month
   int firstSun = 1 + (7 - dow1) % 7;
   bool dst = (d.mon > 3 && d.mon < 11) || (d.mon == 3 && d.day >= firstSun + 7) || (d.mon == 11 && d.day < firstSun);
   return (d.hour - (dst ? 4 : 5) + 24) % 24;
}

bool CloseOwn(const string s, double &fill, double &comm, ulong &deal)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != s || PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      if(!g_trade.PositionClose(ticket)) return false;
      deal = g_trade.ResultDeal();
      fill = g_trade.ResultPrice();
      comm = HistoryDealSelect(deal) ? HistoryDealGetDouble(deal, DEAL_COMMISSION) : 0;
      return true;
   }
   return false;
}

void Row(const int h, const string lab, const string s, const string side, const double vol, const double quote,
         const double fill, const double mid, const bool buy, const long ms, const double comm, const ulong deal)
{
   double slip = (buy ? fill - quote : quote - fill) / mid * 1e4;          // > 0 = worse than the quote
   FileWrite(h, TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS), lab, s, side, DoubleToString(vol, 4),
             DoubleToString(quote, 8), DoubleToString(fill, 8), DoubleToString(slip, 4), ms, DoubleToString(comm, 4), (long)deal);
}

void Probe()
{
   int h = OpenAppend("probes", "utc;lab;symbol;side;volume;quote;fill;slip_bp;exec_ms;commission;deal");
   if(h == INVALID_HANDLE) return;
   for(int i = 0; i < g_n; i++)
   {
      string s = g_sym[i];
      double vol = SymbolInfoDouble(s, SYMBOL_VOLUME_MIN);
      g_trade.SetTypeFillingBySymbol(s);
      for(int side = 0; side < 2; side++)                   // buy then close, sell then close
      {
         MqlTick t; if(!SymbolInfoTick(s, t)) continue;
         bool buy = side == 0;
         ulong t0 = GetMicrosecondCount();
         bool ok = buy ? g_trade.Buy(vol, s) : g_trade.Sell(vol, s);
         long ms = (long)((GetMicrosecondCount() - t0) / 1000);
         if(!ok) { Print("probe failed ", s, " ", g_trade.ResultRetcodeDescription()); continue; }
         ulong deal = g_trade.ResultDeal();
         double comm = HistoryDealSelect(deal) ? HistoryDealGetDouble(deal, DEAL_COMMISSION) : 0;
         Row(h, g_lab[i], s, buy ? "open_buy" : "open_sell", vol, buy ? t.ask : t.bid, g_trade.ResultPrice(),
             (t.ask + t.bid) / 2, buy, ms, comm, deal);
         Sleep(InpProbeHoldSec * 1000);
         MqlTick c; SymbolInfoTick(s, c);
         double fill = 0; ulong cdeal = 0; t0 = GetMicrosecondCount();
         if(CloseOwn(s, fill, comm, cdeal))
            Row(h, g_lab[i], s, buy ? "close_sell" : "close_buy", vol, buy ? c.bid : c.ask, fill, (c.ask + c.bid) / 2,
                !buy, (long)((GetMicrosecondCount() - t0) / 1000), comm, cdeal);
      }
   }
   FileClose(h);
}

//+------------------------------------------------------------------+
void OnTimer()
{
   LogSpreads();
   if(iTime(_Symbol, PERIOD_D1, 0) != g_lastSymbolsDay) LogSymbols();
   MqlDateTime d; TimeToStruct(TimeGMT(), d);
   if(InpProbe && NyHour() == InpProbeHourNY && d.day_of_week >= 1 && d.day_of_week <= 5 && g_lastProbeDay != d.day_of_year)
   {
      g_lastProbeDay = d.day_of_year;
      Probe();
   }
}
//+------------------------------------------------------------------+
