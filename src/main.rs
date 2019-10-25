use std::collections::{ HashMap, HashSet };

fn main(){
    let mut s = Vec::new();
    s.push("abcba".chars().collect::<Vec<char>>());
    s.push("abbac".chars().collect::<Vec<char>>());
    println!("{:?}", s);

    let mut h = HashMap::new();
    let mut hs = HashSet::new();
    let mut last = 0;
    for i in 0..s.len() {
        let mut crt = 0;
        for j in 0..s[i].len() {
            hs.insert(&s[i][j]);
            match h.get(&(crt, s[i][j])) {
                Some(x) => {
                    crt=*x;
                },
                None => {
                    last+=1;
                    h.insert((crt, s[i][j]), last);
                    crt=last;
                }
            }
        }
    }


    println!("{:?}", h);
    println!("{:?}", hs);

    
}