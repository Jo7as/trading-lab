//+------------------------------------------------------------------+
//|                                          TradingLabST_hf_ls01.mq5    |
//|  trading-lab Supertrend+Volume — locked HF baseline (hf_ls01)         |
//|  Port of trading_lab/strategies/supertrend_volume.py          |
//|                                                                  |
//|  Defaults match hf_baseline.yaml:                                |
//|    ST 7/1.5, vol med 20×1.0, EMA off (3/3), SL 1.5 ATR,          |
//|    TP1 2R @ 60%, then Supertrend trail, max hold 48 bars,        |
//|    long+short, 1 position, no pullback                           |
//|                                                                  |
//|  Chart: H1 recommended (NAS100 / US100 / XAUUSD etc.)            |
//+------------------------------------------------------------------+
#property copyright "trading-lab"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

input group "=== hf_ls01 core ==="
input int      InpSTPeriod       = 7;
input double   InpSTMult         = 1.5;
input int      InpVolPeriod      = 20;
input double   InpVolMult        = 1.0;
input int      InpEmaFast        = 3;      // = slow → EMA filter OFF
input int      InpEmaSlow        = 3;
input int      InpAtrPeriod      = 14;
input double   InpMinAtrPct      = 0.0;
input bool     InpPullback       = false;
input bool     InpLongOnly       = false;  // hf_ls01 = long+short

input group "=== Risk / exits ==="
input double   InpRiskPercent    = 2.0;    // % of balance risked per trade (Apex-style)
input double   InpSlAtrMult      = 1.5;
input double   InpTp1R           = 2.0;
input double   InpTp1ClosePct    = 0.60;   // close this fraction at TP1
input int      InpMaxHoldBars    = 48;     // time stop in chart bars
input double   InpMaxDailyLossPct= 3.0;    // pause new entries if day PnL worse
input int      InpMaxPositions   = 1;

input group "=== Misc ==="
input ulong    InpMagic          = 20260720;
input bool     InpOneTradePerDay = true;
input int      InpSlippagePoints = 30;

CTrade         g_trade;
datetime       g_lastBarTime     = 0;
datetime       g_lastTradeDay    = 0;
bool           g_tp1Done         = false;
int            g_entryBarIndex   = -1;
double         g_dayStartEquity  = 0;
int            g_dayStamp        = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePoints);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayStamp = DayStamp();
   Print("TradingLabST_hf_ls01 started on ", _Symbol, " ", EnumToString((ENUM_TIMEFRAMES)_Period),
         " risk=", InpRiskPercent, "%");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {}

void OnTick()
{
   // New calendar day → reset daily equity baseline
   int ds = DayStamp();
   if(ds != g_dayStamp)
   {
      g_dayStamp = ds;
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   }

   ManageOpenPosition();

   // Only act on new closed bar
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == g_lastBarTime)
      return;
   g_lastBarTime = t;

   if(CountOurPositions() >= InpMaxPositions)
      return;

   if(DailyLossHit())
      return;

   if(InpOneTradePerDay && SameDay(g_lastTradeDay, TimeCurrent()))
      return;

   // Need enough history
   int need = MathMax(InpSTPeriod, MathMax(InpVolPeriod, InpAtrPeriod)) + 50;
   if(Bars(_Symbol, _Period) < need)
      return;

   // Evaluate signal on bar 1 (just closed)
   int dir = SignalOnBar(1);
   if(dir == 0)
      return;

   OpenTrade(dir);
}

//+------------------------------------------------------------------+
int DayStamp()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}

bool SameDay(datetime a, datetime b)
{
   if(a <= 0) return false;
   MqlDateTime da, db;
   TimeToStruct(a, da);
   TimeToStruct(b, db);
   return (da.year == db.year && da.mon == db.mon && da.day == db.day);
}

bool DailyLossHit()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (g_dayStartEquity - eq) / MathMax(g_dayStartEquity, 1.0) * 100.0;
   return (dd >= InpMaxDailyLossPct);
}

int CountOurPositions()
{
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      n++;
   }
   return n;
}

//+------------------------------------------------------------------+
//| Wilder ATR at bar index (0=current)                                |
//+------------------------------------------------------------------+
double CalcATR(int period, int shift)
{
   // Build from history: use iATR for simplicity (Wilder-like in MT5)
   int h = iATR(_Symbol, _Period, period);
   if(h == INVALID_HANDLE) return 0;
   double buf[];
   if(CopyBuffer(h, 0, shift, 1, buf) < 1) { IndicatorRelease(h); return 0; }
   IndicatorRelease(h);
   return buf[0];
}

double CalcEMA(int period, int shift)
{
   int h = iMA(_Symbol, _Period, period, 0, MODE_EMA, PRICE_CLOSE);
   if(h == INVALID_HANDLE) return 0;
   double buf[];
   if(CopyBuffer(h, 0, shift, 1, buf) < 1) { IndicatorRelease(h); return 0; }
   IndicatorRelease(h);
   return buf[0];
}

//+------------------------------------------------------------------+
//| Supertrend at bar `shift` — returns line & direction (+1/-1)      |
//+------------------------------------------------------------------+
bool CalcSupertrend(int period, double mult, int shift, double &st_line, double &st_dir)
{
   int bars = Bars(_Symbol, _Period);
   int from = bars - 1;
   int to   = shift;
   if(from - to < period + 5) return false;

   int n = from - to + 1;
   double high[], low[], close[];
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   if(CopyHigh(_Symbol, _Period, to, n, high) < n) return false;
   if(CopyLow(_Symbol, _Period, to, n, low) < n) return false;
   if(CopyClose(_Symbol, _Period, to, n, close) < n) return false;

   // True range + Wilder ATR walking from oldest
   // Convert to chronological arrays for the loop used in Python
   double H[], L[], C[], TR[], ATR[];
   ArrayResize(H, n); ArrayResize(L, n); ArrayResize(C, n);
   ArrayResize(TR, n); ArrayResize(ATR, n);
   for(int i = 0; i < n; i++)
   {
      // series index i corresponds to chrono index n-1-i
      int s = n - 1 - i;
      H[i] = high[s];
      L[i] = low[s];
      C[i] = close[s];
   }
   for(int i = 0; i < n; i++)
   {
      double prevC = (i == 0 ? C[0] : C[i - 1]);
      double tr1 = H[i] - L[i];
      double tr2 = MathAbs(H[i] - prevC);
      double tr3 = MathAbs(L[i] - prevC);
      TR[i] = MathMax(tr1, MathMax(tr2, tr3));
   }
   ATR[0] = TR[0];
   double alpha = 1.0 / period;
   for(int i = 1; i < n; i++)
      ATR[i] = alpha * TR[i] + (1.0 - alpha) * ATR[i - 1];

   double finalUpper[], finalLower[], stLine[], stDir[];
   ArrayResize(finalUpper, n); ArrayResize(finalLower, n);
   ArrayResize(stLine, n); ArrayResize(stDir, n);

   double hl2 = 0.5 * (H[0] + L[0]);
   finalUpper[0] = hl2 + mult * ATR[0];
   finalLower[0] = hl2 - mult * ATR[0];
   stLine[0] = finalUpper[0];
   stDir[0] = (C[0] >= stLine[0] ? 1.0 : -1.0);

   for(int i = 1; i < n; i++)
   {
      hl2 = 0.5 * (H[i] + L[i]);
      double upperBasic = hl2 + mult * ATR[i];
      double lowerBasic = hl2 - mult * ATR[i];

      if(upperBasic < finalUpper[i - 1] || C[i - 1] > finalUpper[i - 1])
         finalUpper[i] = upperBasic;
      else
         finalUpper[i] = finalUpper[i - 1];

      if(lowerBasic > finalLower[i - 1] || C[i - 1] < finalLower[i - 1])
         finalLower[i] = lowerBasic;
      else
         finalLower[i] = finalLower[i - 1];

      if(stDir[i - 1] == 1.0)
      {
         if(C[i] < finalLower[i])
         {
            stDir[i] = -1.0;
            stLine[i] = finalUpper[i];
         }
         else
         {
            stDir[i] = 1.0;
            stLine[i] = finalLower[i];
         }
      }
      else
      {
         if(C[i] > finalUpper[i])
         {
            stDir[i] = 1.0;
            stLine[i] = finalLower[i];
         }
         else
         {
            stDir[i] = -1.0;
            stLine[i] = finalUpper[i];
         }
      }
   }

   st_line = stLine[n - 1];
   st_dir  = stDir[n - 1];
   return true;
}

bool PrevSupertrendDir(int period, double mult, int shift, double &prev_dir)
{
   double line, dir;
   // shift+1 = previous closed bar relative to `shift`
   if(!CalcSupertrend(period, mult, shift + 1, line, dir))
      return false;
   prev_dir = dir;
   return true;
}

//+------------------------------------------------------------------+
double VolumeMedian(int period, int shift)
{
   long vol[];
   ArraySetAsSeries(vol, true);
   if(CopyTickVolume(_Symbol, _Period, shift, period, vol) < period)
      return 0;
   // fallback: if mostly zeros, use range median
   int nonzero = 0;
   for(int i = 0; i < period; i++) if(vol[i] > 0) nonzero++;
   double arr[];
   ArrayResize(arr, period);
   if(nonzero < period / 2)
   {
      double high[], low[];
      ArraySetAsSeries(high, true); ArraySetAsSeries(low, true);
      if(CopyHigh(_Symbol, _Period, shift, period, high) < period) return 0;
      if(CopyLow(_Symbol, _Period, shift, period, low) < period) return 0;
      for(int i = 0; i < period; i++) arr[i] = high[i] - low[i];
   }
   else
   {
      for(int i = 0; i < period; i++) arr[i] = (double)vol[i];
   }
   ArraySort(arr);
   if(period % 2 == 1) return arr[period / 2];
   return 0.5 * (arr[period / 2 - 1] + arr[period / 2]);
}

bool VolumeOk(int shift)
{
   double med = VolumeMedian(InpVolPeriod, shift);
   if(med <= 0) return true;
   long v = iVolume(_Symbol, _Period, shift);
   // if sparse volume, compare range
   double high = iHigh(_Symbol, _Period, shift);
   double low  = iLow(_Symbol, _Period, shift);
   double cur = (v > 0 ? (double)v : (high - low));
   return (cur > med * InpVolMult);
}

//+------------------------------------------------------------------+
//| Signal on closed bar `shift` (1 = last closed). Return +1/-1/0    |
//+------------------------------------------------------------------+
int SignalOnBar(int shift)
{
   double st_line, st_dir, prev_dir;
   if(!CalcSupertrend(InpSTPeriod, InpSTMult, shift, st_line, st_dir))
      return 0;
   if(!PrevSupertrendDir(InpSTPeriod, InpSTMult, shift, prev_dir))
      return 0;

   double close = iClose(_Symbol, _Period, shift);
   double atr = CalcATR(InpAtrPeriod, shift);
   if(atr <= 0) return 0;
   double atr_pct = atr / close;
   if(atr_pct <= InpMinAtrPct) return 0;
   if(!VolumeOk(shift)) return 0;

   bool trend_up = true, trend_down = true;
   if(InpEmaFast != InpEmaSlow)
   {
      double ef = CalcEMA(InpEmaFast, shift);
      double es = CalcEMA(InpEmaSlow, shift);
      trend_up = (ef > es);
      trend_down = (ef < es);
   }

   bool flip_up = (st_dir > 0 && prev_dir < 0);
   bool flip_dn = (st_dir < 0 && prev_dir > 0);

   if(flip_up && close > st_line && trend_up)
      return +1;
   if(!InpLongOnly && flip_dn && close < st_line && trend_down)
      return -1;

   // pullback optional (off for hf_ls01)
   if(InpPullback)
   {
      double low = iLow(_Symbol, _Period, shift);
      double high = iHigh(_Symbol, _Period, shift);
      double tol = 1.0 + 0.002;
      if(st_dir > 0 && prev_dir > 0 && low <= st_line * tol && close > st_line && !flip_up && trend_up)
         return +1;
      if(!InpLongOnly && st_dir < 0 && prev_dir < 0 && high >= st_line / tol && close < st_line && !flip_dn && trend_down)
         return -1;
   }
   return 0;
}

//+------------------------------------------------------------------+
double NormalizeLots(double lots)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0) step = 0.01;
   lots = MathFloor(lots / step) * step;
   if(lots < minLot) lots = minLot;
   if(lots > maxLot) lots = maxLot;
   return lots;
}

double LotsForRisk(double entry, double sl)
{
   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tickSize <= 0 || tickValue <= 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double slDist = MathAbs(entry - sl);
   if(slDist <= 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double ticks = slDist / tickSize;
   double lots = riskMoney / (ticks * tickValue);
   return NormalizeLots(lots);
}

void OpenTrade(int dir)
{
   double atr = CalcATR(InpAtrPeriod, 1);
   if(atr <= 0) return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double entry, sl, tp;
   ENUM_ORDER_TYPE typ;

   if(dir > 0)
   {
      entry = ask;
      sl = entry - InpSlAtrMult * atr;
      tp = entry + InpTp1R * (entry - sl);
      typ = ORDER_TYPE_BUY;
   }
   else
   {
      entry = bid;
      sl = entry + InpSlAtrMult * atr;
      tp = entry - InpTp1R * (sl - entry);
      typ = ORDER_TYPE_SELL;
   }

   double lots = LotsForRisk(entry, sl);
   // Open full size; we partial-close at TP1 in ManageOpenPosition
   // Set broker SL to initial SL; TP left empty (we manage TP1 partially)
   bool ok;
   if(typ == ORDER_TYPE_BUY)
      ok = g_trade.Buy(lots, _Symbol, entry, sl, 0.0, "TradingLabST hf_ls01 L");
   else
      ok = g_trade.Sell(lots, _Symbol, entry, sl, 0.0, "TradingLabST hf_ls01 S");

   if(ok)
   {
      g_tp1Done = false;
      g_entryBarIndex = Bars(_Symbol, _Period);
      g_lastTradeDay = TimeCurrent();
      // Store TP1 in global via comment on position — use a file/global var
      GlobalVariableSet("TradingLabST_TP1_" + IntegerToString(InpMagic), tp);
      GlobalVariableSet("TradingLabST_ENTRY_" + IntegerToString(InpMagic), entry);
      GlobalVariableSet("TradingLabST_LOTS_" + IntegerToString(InpMagic), lots);
      Print("Opened ", (dir > 0 ? "LONG" : "SHORT"), " lots=", lots, " SL=", sl, " TP1=", tp);
   }
   else
      Print("Order failed: ", g_trade.ResultRetcode(), " ", g_trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
void ManageOpenPosition()
{
   if(CountOurPositions() == 0)
      return;

   ulong ticket = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      ticket = t;
      break;
   }
   if(ticket == 0) return;

   long ptype = PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl = PositionGetDouble(POSITION_SL);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price = (ptype == POSITION_TYPE_BUY ? bid : ask);

   string gTp = "TradingLabST_TP1_" + IntegerToString(InpMagic);
   string gLots = "TradingLabST_LOTS_" + IntegerToString(InpMagic);
   double tp1 = GlobalVariableGet(gTp);
   double openLots = GlobalVariableGet(gLots);
   if(openLots <= 0) openLots = volume;

   // Time stop
   int barsHeld = Bars(_Symbol, _Period) - g_entryBarIndex;
   if(g_entryBarIndex > 0 && barsHeld >= InpMaxHoldBars)
   {
      g_trade.PositionClose(ticket);
      Print("Time stop after ", barsHeld, " bars");
      return;
   }

   // TP1 partial
   if(!g_tp1Done && tp1 > 0)
   {
      bool hit = false;
      if(ptype == POSITION_TYPE_BUY && bid >= tp1) hit = true;
      if(ptype == POSITION_TYPE_SELL && ask <= tp1) hit = true;
      if(hit)
      {
         double closeVol = NormalizeLots(openLots * InpTp1ClosePct);
         double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         if(closeVol >= minLot && closeVol < volume)
         {
            if(g_trade.PositionClosePartial(ticket, closeVol))
            {
               g_tp1Done = true;
               // Move SL to breakeven
               double be = openPrice;
               g_trade.PositionModify(ticket, be, 0);
               Print("TP1 partial ", closeVol, " BE SL=", be);
            }
         }
         else
         {
            // close all if partial too small
            g_trade.PositionClose(ticket);
            g_tp1Done = true;
            Print("TP1 full close (partial size too small)");
            return;
         }
      }
   }

   // After TP1: Supertrend trail + flip exit (check on new bar ideally; also tick)
   if(g_tp1Done)
   {
      double st_line, st_dir;
      if(CalcSupertrend(InpSTPeriod, InpSTMult, 0, st_line, st_dir))
      {
         if(ptype == POSITION_TYPE_BUY)
         {
            double newSL = MathMax(sl, st_line);
            if(newSL > sl + _Point)
               g_trade.PositionModify(ticket, newSL, 0);
            if(st_dir < 0)
            {
               g_trade.PositionClose(ticket);
               Print("Supertrend flip exit (long)");
               return;
            }
         }
         else
         {
            double newSL = (sl <= 0 ? st_line : MathMin(sl, st_line));
            if(sl <= 0 || newSL < sl - _Point)
               g_trade.PositionModify(ticket, newSL, 0);
            if(st_dir > 0)
            {
               g_trade.PositionClose(ticket);
               Print("Supertrend flip exit (short)");
               return;
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
