#include <bits/stdc++.h>

using namespace std;

int main() {
    vector<int> v = {1, 2, 3, 3};

    cout << "v.begin() = " << *v.begin() << '\n';

    for (int i = 0; i <= 4; i++ ) {
        auto it1 = upper_bound(v.begin(), v.end(), i);
        cout << *it1 << " -> " << *prev(it1) << '\n';
        cout << " ------ " << (it1 != v.begin()) << '\n';
    }
}