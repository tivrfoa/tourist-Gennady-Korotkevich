/**
 *    author:  tourist
 *    created: 03.01.2023 20:21:48       
**/
#include <bits/stdc++.h>

using namespace std;

#ifdef LOCAL
#include "algo/debug.h"
#else
#define debug(...) 42
#endif

long long Q = 0;
bool Win(vector<long long> a, int k, long long s) {
  ++Q;
//  debug(a, k, s);
  int n = (int) a.size();
  if (n == k) {
    long long opp = accumulate(a.begin(), a.end(), 0LL);
    return (s >= opp);
  }
  sort(a.rbegin(), a.rend());
  if (k == 1) {
    for (int i = 0; i < n; i++) {
      if (s >= a[i]) {
        auto b = a;
        b.erase(b.begin() + i);
        return Win(b, 2, s + a[i]);
      }
    }
    return false;
  }
  if (k == 2) {
    int mn = n;
    for (int i = 0; i < n; i++) {
      int val = n;
      for (int j = i + 1; j < n; j++) {
        if (s >= a[i] + a[j]) {
          val = j;
          break;
        }
      }
      if (val < mn) {
        mn = val;
        auto b = a;
        b.erase(b.begin() + val);
        b.erase(b.begin() + i);
        if (Win(b, 4, s + a[i] + a[val])) {
          return true;
        }
      }
    }
    return false;
  }
  if (k == 4) {
    vector<vector<vector<int>>> mn(n, vector<vector<int>>(n, vector<int>(n, n)));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          int val = n;
          if (i < j && j < k) {
            for (int t = k + 1; t < n; t++) {
              if (s >= a[i] + a[j] + a[k] + a[t]) {
                val = t;
                break;
              }
            }
          }
          mn[i][j][k] = n;
          if (i > 0) mn[i][j][k] = min(mn[i][j][k], mn[i - 1][j][k]);
          if (j > 0) mn[i][j][k] = min(mn[i][j][k], mn[i][j - 1][k]);
          if (k > 0) mn[i][j][k] = min(mn[i][j][k], mn[i][j][k - 1]);
          if (val < mn[i][j][k]) {
            mn[i][j][k] = val;
            auto b = a;
            b.erase(b.begin() + val);
            b.erase(b.begin() + k);
            b.erase(b.begin() + j);
            b.erase(b.begin() + i);
            if (Win(b, 8, s + a[i] + a[j] + a[k] + a[val])) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }
  if (k == 8) {
    assert(n == 24);
    long long total = accumulate(a.begin(), a.end(), 0LL) + s;
    long long goal = (total + 1) / 2;
    if (2 * s < goal) {
      return false;
    }
    vector<vector<long long>> all(9);
    bool found = false;
    function<void(int, int, long long)> Dfs = [&](int v, int w, long long sum) {
      if (found) {
        return;
      }
      if (v == 12) {
        if (sum <= s) {
          if (s + sum >= goal) {
            found = true;
          } else {
            all[w].push_back(sum);
          }
        }
        return;
      }
      if (w < 8) {
        Dfs(v + 1, w + 1, sum + a[v]);
      }
      Dfs(v + 1, w, sum);
    };
    Dfs(0, 0, 0);
    if (found) {
      return true;
    }
    for (int i = 0; i <= 8; i++) {
      sort(all[i].begin(), all[i].end());
    }
    function<void(int, int, long long)> Find = [&](int v, int w, long long sum) {
      if (found) {
        return;
      }
      if (v == 24) {
        auto& vec = all[8 - w];
        auto it = upper_bound(vec.begin(), vec.end(), s - sum);
        if (it != vec.begin()) {
          sum += *prev(it);
          if (s + sum >= goal) {
            found = true;
          }
        }
        return;
      }
      if (w < 8) {
        Find(v + 1, w + 1, sum + a[v]);
      }
      Find(v + 1, w, sum);
    };
    Find(12, 0, 0);
    return found;
  }
  assert(false);
  return false;
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  int n;
  cin >> n;
  vector<long long> a(n);
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  vector<int> order(n);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int i, int j) {
    return a[i] < a[j];
  });
  int low = -1;
  int high = n - 1;
  while (low + 1 < high) {
    int mid = (low + high) >> 1;
    auto b = a;
    b.erase(b.begin() + order[mid]);
    if (Win(b, 1, a[order[mid]])) {
      high = mid;
    } else {
      low = mid;
    }
  }
  string res(n, '0');
  for (int i = high; i < n; i++) {
    res[order[i]] = '1';
  }
  cout << res << '\n';
  debug(Q, clock());
  return 0;
}
