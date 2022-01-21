import Foundation

/*
一般写算法用c语言来实现，比较能理解整个细节，因为高级语言都封装的简易的操作，
像数组是动态的，queue需要维护front和rear指针，stack要维护top指针，
c语言是面向过程的，所以在实现字符串操作相对容易，但在没有内置的map数据类型，
所以像实现LRU Cache用到双向链表和hashmap来实现，代码比较长
*/

//MARK: 53. 最大子数组和
//动态转移方程，就是递推公式，当前状态由上次状态转移，有时可以优化用滚动数组，即用一个变量来维护上一个结果

func maxSubArray(_ nums: [Int]) -> Int {
    //f(n) = max{f(n)+n, n}
    var pre = 0
    var ans = nums[0]
    for n in nums {
        pre = max(pre + n, n)
        ans = max(pre, ans)
    }
    return ans
}
//MARK: 206. 反转链表
func reverseList(_ head: ListNode?) -> ListNode? {
    //头插法
    var pre: ListNode? = nil
    var cur = head
    while cur != nil {
        var next = cur?.next //取出next指针，因为要改变指针方向
        cur?.next = pre  //翻转操作
        pre = cur  //移动cur和next指针
        cur = next
    }
    return pre
}
//递归法
func reverseList(_ head: ListNode?) -> ListNode? {
    //递归出口，当head.next为空，到链表最后一个。head == nil是判断传的是个nil
    if head == nil || head?.next == nil {
        return head
    }
    var newHead = reverseList(head?.next)
    head?.next?.next = head  //这个里做翻转操作
    head?.next = nil //这里防止出现循环链表，因为最后一个要为nil
    return newHead
}

//MARK: 110. 平衡二叉树
func isBalanced(_ root: TreeNode?) -> Bool {
    if (root == nil) {
        return true
    } else {
        //三个条件左右相差1，并且左子树和右子树都平衡
        return abs(height(root?.left) - height(root?.right)) <= 1 && isBalanced(root?.left) && isBalanced(root?.right)
    }
}
//辅助函数计算高度
func height(_ root: TreeNode?) -> Int {
    if root == nil {
        return 0
    } else {
        return max(height(root?.left), height(root?.right)) + 1 //因为高度从1开始计算
    }
}

/*
 1 2 3
 4 5 6
 7 8 9
 按层模拟: 123, 69, 8, 47, 5
 回字形一层一层遍历
 */

func spiralOrder(_ matrix: [[Int]]) -> [Int] {
    guard !matrix.isEmpty else { return [] }
    var l = 0
    var r = matrix[0].count - 1
    var t = 0
    var b = matrix.count - 1
    var res = [Int]()
    while (l <= r && t <= b) {
        for i in l...r {
            res.append(matrix[t][i]) //把第一行加入到结果数组
        }
        //range注意上下界, 或者用stride(from:to:by:)不会有问题, 其他语言是用for来不会有这个问题
        if t+1 > b { break }
        for i in t+1...b { //上指针下移一个位置
            res.append(matrix[i][r]) //把最后一列除了第一个，加入到结果数组
        }
        // 边界缩减了1，所以判断条件不能<=
        if (l < r && t < b) { //因为边界在++或者--, 右到左，下到上边界减少1
            for i in (l+1..<r).reversed() { //左右缩减1
                res.append(matrix[b][i]) //列在变
            }
            for i in (t+1...b).reversed() {
                res.append(matrix[i][l]) //行在变
            }
        }
        //然后下一层
        l += 1
        r -= 1
        t += 1
        b -= 1
    }
    return res
}
/*
 1 2 3
 4 5 6
 7 8 9
 */

//转圈遍历  先 123, 69, 87, 5
func spiralOrder(_ matrix: [[Int]]) -> [Int] {
    guard !matrix.isEmpty else { return [] }
    var l = 0
    var r = matrix[0].count - 1
    var t = 0
    var b = matrix.count - 1
    var n = matrix[0].count * matrix.count
    var res = [Int]()
    while res.count < n {
        //to的元素不包含，如果使用range注意上下界， 例如：l...r l > r 会崩溃
        for i in stride(from: l, to: r+1, by: 1) where res.count < n {
            res.append(matrix[t][i])
        }
        t += 1
        for i in stride(from: t, to: b+1, by: 1) where res.count < n {
            res.append(matrix[i][r])
        }
        r -= 1
        for i in stride(from: r, to: l-1, by: -1) where res.count < n {
            res.append(matrix[b][i])
        }
        b -= 1
        for i in stride(from: b, to: t-1, by: -1) where res.count < n {
            res.append(matrix[i][l])
        }
        l += 1
    }
    return res
}

//MARK: 146. LRU 缓存
//哈希表+双向链表
// 靠近头部最近使用，靠近尾部最久未使用
//通过hashtable实现O(1)插入和查询
/*
 cache: 1 : 1
        2 : 2
 
 head -> 2 -> 1 -> tail
      <-   <-   <-
 
 因为用c语言实现hashtable代码量大，可以使用c语言优秀库uthash底层本身就是用双向链表实现的hash来实现
 */
class DLinkedNode {
    var key: Int
    var value: Int
    var pre: DLinkedNode?
    var next: DLinkedNode?
    init(_ key: Int, _ val: Int) {
        self.key = key
        self.value = val
    }
}

class LRUCache {
    var cache = [Int: DLinkedNode]()
    var count: Int = 0
    var capacity: Int
    let head = DLinkedNode(0, 0)
    let tail = DLinkedNode(0, 0)
    
    init(_ capacity: Int) {
        self.capacity = capacity
        head.next = tail
        tail.pre = head
    }
    func remove(_ key: Int) {
        guard count > 0, let node = cache[key] else {
            return
        }
        cache[key] = nil
        //双向断开连接
        node.pre?.next = node.next
        node.next?.pre = node.pre
        node.pre = nil
        node.next = nil
        count -= 1
    }
    //头插法
    func insert(_ node: DLinkedNode) {
        cache[node.key] = node
        //在头部插入
        //head的下一个连接node
        node.next = head.next
        head.next?.pre = node
        //node连接head
        node.pre = head
        head.next = node
        count += 1
    }
    
    func get(_ key: Int) -> Int {
        if let node = cache[key] {
            // 删除并插入，相当于移动到头部
            remove(key)
            insert(node)
            return node.value
        }
        return -1
    }
    func put(_ key: Int, _ value: Int) {
        if let node = cache[key] {
            //如果存在，更新值，并移动到头部
            node.value = value
            remove(key)
            insert(node)
            return
        }
        let node = DLinkedNode(key, value)
        cache[key] = node
        //如果已经满了，移除最后一个node
        if count == capacity, let tailKey = tail.pre?.key {
            remove(tailKey)
        }
        //插入到头部
        insert(node)
    }
}


//MARK: 102. 二叉树的层序遍历
func levelOrder(_ root: TreeNode?) -> [[Int]] {
    guard let root = root else {
        return []
    }
    var res = [[Int]]()
    var queue: [TreeNode] = [root] //因为上面做了guard守护
    while !queue.isEmpty {
        var level = [Int]()
        for _ in queue {  //只要遍历当前队列，不需要i，拓展当前层节点的下层
            var node = queue.removeFirst() //出队，用数组模拟
            level.append(node.val)
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
        res.append(level)
    }
    return res
}

class MyQueue {
    var inStack = [Int]()
    var outStack = [Int]()
    // var front = -1 //也可以用一个属性记录队列头元素

    init() {

    }
    
    func push(_ x: Int) {
    //    if inStack.isEmpty {
    //        front = x
    //    }
        inStack.append(x)
    }
    
    func pop() -> Int {
        if (outStack.isEmpty) {
            while(!inStack.isEmpty) {
                outStack.append(inStack.removeLast())
            }
        }
        return outStack.popLast() ?? -1
    }
    
    func peek() -> Int {
        // return outStack.last ?? inStack.last ?? -1
        return outStack.last ?? inStack.first ?? -1 //不用变量就使用这种方式
    }
    
    func empty() -> Bool {
        return inStack.isEmpty && outStack.isEmpty
    }
}

//MARK: 69. Sqrt(x)
func mySqrt(_ x: Int) -> Int {
    var l = 0
    var r = x
    var ans = -1
    while l <= r {
        let mid = l + (r - l) / 2
        if mid * mid <= x {
            ans = mid
            l = mid + 1 //在右边
        } else {
            r = mid - 1 //在左边
        }
    }
    return ans
}
//如果精确到多少位 swift有类型判断
func mySqrt(_ x: Double, _ epsilon: Double) -> Double {
    var l = 0
    var r = x
    if (x == 0 || x == 1) {
        return x
    }
    while l < r {
        let mid = l + (r - l) / 2
        if (abs(mid * mid) - x < epsilon) {
            return mid
        } else if (mid * mid < x) {
            l = mid
        } else {
            r = mid
        }
    }
    return l
}


//牛顿迭代法 等价于 求 f(x) = x^2 - a 的正根， 因为f'(x) = 2x, 根据斜截式求出x轴交点为  x - f(x)/2x , f(x)代入得
//（x + a/x) / 2

func mySqrt(_ x: Int) -> Int {
    var ans = x
    while (ans * ans > x) {
        ans = (ans + x / ans) / 2
    }
    return ans
}


//MARK: 排序数组，归并排序
func sortArray(_ nums: [Int]) -> [Int] {
    guard nums.count > 1 else {return nums}
    
    let m = nums.count / 2
    //左右递归
    let l = sortArray(Array(nums[0..<m]))
    let r = sortArray(Array(nums[m..<nums.count]))
    //然后再合并
    return merge(lPie: l, rPie: r)

}

//重点是合并算法，子问题就是分区，然后子问题递归解决大问题，先递归后合并，然后不是原地排序，需要额外临时数组
func merge(lPie: [Int], rPie: [Int]) -> [Int] {

    var l = 0
    var r = 0
    var a = [Int]()
    
    while l < lPie.count && r < rPie.count {
        if lPie[l] < rPie[r] {
            a.append(lPie[l])
            l += 1
        } else {
            a.append(rPie[r])
            r += 1
        }
    }
    //还有剩余情况
    while l < lPie.count {
        a.append(lPie[l])
        l += 1
    }
    
    while r < rPie.count {
        a.append(rPie[r])
        r += 1
    }
    
    return a
}

//MARK: 快排
func sortArray(_ nums: [Int]) -> [Int] {
    var arr = nums;
    quickSort(&arr, 0, arr.count - 1)
    return arr
}
//重点是分区算法，子问题就是分区，然后子问题递归解决大问题，先分区后递归，是原地排序
func quickSort(_ nums: inout [Int], _ l: Int, _ r: Int) {
    if (l > r) {
        return
    }
    //随机一个数交换, 因为如果原来有序的数组会退化为O(n^2)
    let random = Int.random(in: l...r)
    (nums[random], nums[r]) = (nums[r], nums[random])

    //下面做分区操作，去r位置的元素作为基准值
    var x = nums[r]
    // 定义i和j指针，分为三个区域,0..<i,为小区，i..<j为大区，j..<r为未查看
    var i = l
    for j in l..<r {
        if (nums[j] <= x) {
            (nums[j], nums[i]) = (nums[i], nums[j])
            i += 1
        }
    }
    //把基准的元素放到i的位置
    (nums[r], nums[i]) = (nums[i], nums[r])
    //递归左右区间
    quickSort(&nums, l, i - 1)
    quickSort(&nums, i + 1, r)
}

//MARK: 41. 缺失的第一个正数
// 哈希表法：没有出现的最小正整数在[1, N+1]，要么[1, N]或者 N + 1
func firstMissingPositive(_ nums: [Int]) -> Int {
    var nums = nums
    //将所有负数改为n+1
    let n = nums.count
    for i in 0..<n {
        if (nums[i] <= 0) {
            nums[i] = n + 1
        }
    }
    // 将小于等于n的元素位置变为负数
    for i in 0..<n {
        let index = abs(nums[i])
        if (index <= n) {
            nums[index - 1] = -abs(nums[index - 1])
        }
    }
    //找出第一个大于0的元素，下标+1
    for i in 0..<n {
        if (nums[i] > 0) {
            return i + 1
        }
    }
    return n + 1
}

//swift不好操作字符串，api很繁琐，不好获取下标
/*
let name = "King"
let arr = Array(name)
一般采取转成数组来操作
*/
//MARK: 14. 最长公共前缀
//参考leetcode国际站，最简洁的代码
//纵向每个字符串对比
func longestCommonPrefix(_ strs: [String]) -> String {
    if strs.isEmpty { return "" }
    var common = strs[0]
    
    for ch in strs {
        while !ch.hasPrefix(common) {
            common = String(common.dropLast())
        }
    }
    return common
    
}

//MARK: 160. 相交链表
/*
 a单独节点个数为a, b单独节点个数为b, a和b共同节点为c
 a + c + b = b + c + a
 双指针把a走完，然后移动到b
 把b走完，然后移动到a
 然后在交点交汇
*/
func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
    if headA == nil || headB == nil {
        return nil
    }
    var a: ListNode? = headA
    var b: ListNode? = headB
    while a !== b { //比较两个指针需要用!== , !=用来比较值 
        a = a == nil ? headB : a?.next
        b = b == nil ? headA : b?.next
    }
    return a
}

//MARK: 剑指 Offer 31. 栈的压入、弹出序列
//用一个栈模拟弹栈，栈顶元素 == 弹出序列的当前元素，如果栈为空则弹出顺序合法

func validateStackSequences(_ pushed: [Int], _ popped: [Int]) -> Bool {
    var stack = [Int]()
    var i = 0
    for num in pushed {
        stack.append(num) 
        while !stack.isEmpty && stack.last == popped[i] {
            stack.removeLast()
            i += 1
        }
    }
    return stack.isEmpty
}

//MARK: 141. 环形链表
//快慢指针，慢指针走一步，快指针走两步
//注意慢指针在head, 快指针在head.next，因为这样才能while走起来，
//相当于第一次都在虚节点出发，然后慢指针走一步到head，快指针走两步到head.next
func hasCycle(_ head: ListNode?) -> Bool {
    if (head == nil || head?.next == nil) {
        return false
    }
    var slow = head
    var fast = head?.next
    while (fast !== slow) {
        //fast == nil只判断这个也行，这个是为了节点单数，或者双数节点，链表走完
        if (fast == nil || fast?.next == nil) {
            return false
        }
        slow = slow?.next
        fast = fast?.next?.next
    }
    return true
}

//MARK: 226. 翻转二叉树

func invertTree(_ root: TreeNode?) -> TreeNode? {
    if (root == nil) {
        return nil
    }
    let left = invertTree(root?.left)
    let right = invertTree(root?.right)
    root?.left = right
    root?.right = left
    return root
}


//MARK: 704. 二分查找

func search(_ nums: [Int], _ target: Int) -> Int {
    var l = 0
    var r = nums.count - 1
    while(l <= r) {
        let mid = (r - l) / 2 + l
        let num = nums[mid]
        if (num == target) {
            return mid
        } else if (num > target) {
            r = mid - 1
        } else {
            l = mid + 1
        }
    }
    return -1
}

//如果右边界是取不到的， r不能 mid - 1，然后循环条件是 l < r
func search(_ nums: [Int], _ target: Int) -> Int {
    var l = 0
    var r = nums.count
    while l < r {
        let mid = (l + r) / 2
        if nums[mid] == target {
            return mid
        } else if (nums[mid] < target) {
            l = mid + 1
        } else {
            r = mid
        }
    }
    return -1
}


//MARK: 2. 两数相加
//因为逆序相加，然后返回也是逆序，所以直接对应位置相加，然后维护一个进位carry

//因为要返回一个链表，所以需要维护一个head指针，尾指针tail用来拼接链表

//国际站优秀递归解法
class Solution {
    private var anchor = 0
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        if l1 == nil && l2 == nil && anchor == 0 { return nil }
        let sum = (l1?.val ?? 0) + (l2?.val ?? 0) + anchor
        anchor = sum / 10
        let node: ListNode? = ListNode(sum % 10, addTwoNumbers(l1?.next, l2?.next))
        return node
    }
}

func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    var l1: ListNode? = l1
    var l2: ListNode? = l2
    
    var tail: ListNode? = ListNode(0) //head和tail在虚拟头节点
    let head = tail
    
    var carry = 0
    while l1 != nil || l2 != nil || carry > 0 {
        let n1 = l1?.val ?? 0
        let n2 = l2?.val ?? 0
        let sum = n1 + n2 + carry
        carry = sum / 10
        tail?.next = ListNode(sum % 10)
        tail = tail?.next
        l1 = l1?.next
        l2 = l2?.next
    }
    
    return head?.next //返回下个节点，head是虚拟头
}

//MARK: 剑指 Offer 54. 二叉搜索树的第k大节点

//利用swift函数可以嵌套定义，避免用属性记录值
//利用中序遍历逆序，然后返回第k大的元素
class Solution {
    func kthLargest(_ root: TreeNode?, _ k: Int) -> Int {
        var count = 0
        var res = 0
        func dfs(_ root: TreeNode?) {
            guard let root = root else { return }
            dfs(root.right)
            count += 1
            if count == k {
                res = root.val
                return
            }
            dfs(root.left)
        }
        dfs(root)
        return res
    }
}

//中序遍历顺序，然后返回第k个元素，但需要额外空间O(n)
class Solution {
    func inorder(_ root: TreeNode?, _ res: inout [Int]) {
        guard let node = root else { return }
        inorder(node.left, &res)
        res.append(node.val)
        inorder(node.right, &res)
    }
    func kthLargest(_ root: TreeNode?, _ k: Int) -> Int {
        var res: [Int] = []
        inorder(root, &res)
        return res[res.count - k]
    }
}

//MARK: 142. 环形链表 II  返回第一个入环节点
//根据归纳计算得出 a = c + (n-1)(b+c)  fast和slow指针相遇时，slow指针和从起点出发的指针ptr在入口相遇

func detectCycle(_ head: ListNode?) -> ListNode? {
    var fast = head
    var slow = head
    while(true) { //fast跑圈，因为可能要跑n圈才能交汇
        // fast?.next 表示刚好在尾节点，下个节点为空，少循环一次
        if (fast == nil || fast?.next == nil) {
            return nil
        }
        fast = fast?.next?.next
        slow = slow?.next
        if (fast == slow)  {
            break
        }
    }
    //重复利用fast指针
    fast = head
    while (slow != fast) {
        slow = slow?.next
        fast = fast?.next
    }
    return fast
}

//简洁版代码
func detectCycle(_ head: ListNode?) -> ListNode? {
  var fast = head
  var slow = head
  while fast != nil {
    slow = slow?.next
    fast = fast?.next?.next
    if fast === slow {
      fast = head
      while fast !== slow {
        fast = fast?.next
        slow = slow?.next
      }
      return fast
    }
  }
  return nil
}

//MARK: 3. 无重复字符的最长子串
//滑动窗口,用set来判断是否有重复字符

//用string的实例方法 contains比较慢
//可以用dict来优化查找速度，因为用set操作不方便

func lengthOfLongestSubstring(_ s: String) -> Int {
    var map = [Character:Int]() //字典来跟踪字母的最后一次出现的索引
    var result = 0  //记录结果
    var l = -1 //左边界初始值
    for (r, char) in s.enumerated(){
        l = max(l, map[char] ?? -1) //更新左边界
        result = max(result, r - l) //计算右减左
        map[char] = r //更新字母最新的索引
    }
    return result

}

//这个运行速度差不多
func lengthOfLongestSubstring2(_ s: String) -> Int {
    var maxCount = 0
    var array = [Character]()
    for char in s {
        if let index = array.firstIndex(of: char) {
            //从0到i都删除掉, 不包括i,因为要求连续的子串
            array.removeFirst(index+1) //更新左边界
        }
        array.append(char)// 更新右边界
        maxCount = max(maxCount, array.count)
    }
    return maxCount
}


//MARK: 同构字符串
func isIsomorphic(_ s: String, _ t: String) -> Bool {
    var s2t: [Character: Character] = [:]
    var t2s: [Character: Character] = [:]

    for (i,_) in s.enumerated() {
        // 这种字符串取下标，会导致时间过长。通过转换成字符串数组解决
        let x = s[s.index(s.startIndex, offsetBy: i)]
        let y = t[t.index(t.startIndex, offsetBy: i)]
        if (s2t[x] != nil && s2t[x] != y) || (t2s[y] != nil && t2s[y] != x){
            return false
        }
        s2t[x] = y
        t2s[y] = x
    }
    return true
}

//通过转成字符数组来解决分割数组问题
func isIsomorphic1(_ s: String, _ t: String) -> Bool {
    let sChars = Array(s)
    let tChars = Array(t)
    var s2t = [Character: Character]()
    
    for i in 0 ..< sChars.count {
        if let char = s2t[sChars[i]] {
            if char != tChars[i] {
                return false
            }
        } else {
            if s2t.values.contains(tChars[i]) {
                return false
            }
            s2t[sChars[i]] = tChars[i]
        }
    }
    return true
}


//MARK: 1. 两数之和
//利用map记录差值 key为数组值， value为数组下标index
func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    var dict = [Int: Int]()
    for (i, num) in nums.enumerated() {
        if (dict[target - num] != nil) {
            return [dict[target - num]!, i]
        }
        dict[num] = i
    }
    return []
}

//MARK: 合并两个有序链表

//递归法，简洁加上可选绑定简化了解包
func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
    guard let list1 = list1 else {return list2}
    guard let list2 = list2 else {return list1}
    if list1.val < list2.val {
        list1.next = mergeTwoLists(list1.next, list2)
        return list1
    } else {
        list2.next = mergeTwoLists(list1, list2.next)
        return list2
    }
}
//双指针

func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
    
    var l1 = list1
    var l2 = list2
    var newHead: ListNode? = ListNode(-1) //假的头结点 注意这里的类型为option
    var pre = newHead
    while (l1 != nil && l2 != nil) {
        if l1!.val <= l2!.val {
            pre?.next = l1
            l1 = l1?.next
        } else {
            pre?.next = l2
            l2 = l2?.next
        }
        pre = pre?.next
    }
    pre?.next = l1 ?? l2
    return newHead?.next
}

//MARK: 300. 最长递增子序列

// dp[i] = max{ dp[j] } + 1    j in 0..<i

func lengthOfLIS(_ nums: [Int]) -> Int {
    
    var dp = [Int]()
    //初始化dp数组为1，因为不像其他语言动态赋值 dp[i] = 1
    dp = (0..<nums.count).map{_ in return 1 }
    var ans = 1
    for i in 1..<nums.count {
        for j in 0..<i {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j] + 1)
            }
        }
        ans = max(ans, dp[i])
    }
    return ans
}

//方法二： 贪心算法 + 二分查找，优化为nlogn
//贪心：如果要上升子序列尽可能小，那每次上升子序列最后加上的数尽可能小
/*
1. 如果 nums[i] > d[len], 则直接加入数组末尾，并更新len += 1;
2. 否则，在d数组中二分查找，找到第一个比nums[i]小的数d[k],并更新d[k+1] = nums[i];
 以输入序列 [0, 8, 4, 12, 2] 为例
 第一步插入 0，d = [0]
 第二步插入 8，d = [0, 8]
 第三步插入 4，d = [0, 4]
 第四步插入 12，d = [0, 4, 12]
 第五步插入 2，d = [0, 2, 12]
 */

func lengthOfLIS(_ nums: [Int]) -> Int {
    // 有三种便捷初始化数组
     var tails = (0..<nums.count).map {_ in return 0}
    // var tails = [Int](repeating: 0, count: nums.count)
    //注意这会初始化为这个 0 --- nums.count
//    var tails = Array(0..<nums.count)
    var res = 0
    for num in nums {
        var l = 0
        var r = res //这里不用减1，因为有可能是插入数组末尾
        while l < r {
            var m = (l + r) / 2
            if (tails[m] < num) {
                l = m + 1
            } else {
                r = m
            }
        }
        //二分查找第一个 num比数组元素小的替换
        // l或者r都可以
        tails[l] = num
        //如果nums[i] > d[len], 则直接加入数组末尾，并更新len += 1
        if (res == r) {
            res += 1
        }
    }
    return res
}


//MARK: 415. 字符串相加

//模拟相加,字符串转成字符数组比较好处理

func addStrings(_ num1: String, _ num2: String) -> String {
    let s1 = Array(num1)
    let s2 = Array(num2)
    var i = s1.count - 1
    var j = s2.count - 1
    var add = 0
    var ans = [Int]()
    //当最后一位还有进位时，还要进入循环，把进位加上去
    // 例如 1 + 9 的情况
    while (i >= 0 || j >= 0 || add != 0) {
        let x = i >= 0 ? Int(String(s1[i]))! - 0 : 0
        let y = j >= 0 ? Int(String(s2[j]))! - 0 : 0
        let res = x + y + add
        ans.append(res % 10)
        add = res / 10
        i -= 1
        j -= 1
    }
    //或者当最后一位还有进位时，后面补1
    var str = ""
    for s in ans.reversed() {
        str += "\(s)"
    }
    return str
}

//MARK: 227. 基本计算器 II
//s 由整数和算符 ('+', '-', '*', '/') 组成，中间由一些空格隔开
//思路：利用栈保存乘除的值，遇到乘除与栈顶元素计算并更新栈顶元素，这样可以优先计算乘除，然后处理完乘除后再把栈的元素相加起来（如果是减就加上负数）

func calculate(_ s: String) -> Int {
    var stack = [Int]()
    var num = 0
    var op = "+"
    let count = s.count //这里要把count提前计算，不然会超时，因为count会每次遍历一次才能得到
    for (i,c) in s.enumerated() {
        //因为字符可以是连续的，比如123
        if c.isNumber {
            num = num * 10 + Int(String(c))!
        }
        //判断i == count - 1处理最后一个字符
        //注意op为记录当前数字的前一个操作符
        if !c.isNumber && c != " " || i == count - 1 {
            switch op {
            case "+":
                stack.append(num)
            case "-":
                stack.append(-num)
            case "*":
                stack.append(stack.removeLast() * num)
            default:
                stack.append(stack.removeLast() / num)
            }
            num = 0  //重置下num
            op = String(c) //操作符要更新一次
        }
    }
    return stack.reduce(0, +)
}

//MARK: 215. 数组中的第K个最大元素
func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
    var nums = nums
    return quickSelect(&nums, 0, nums.count - 1, nums.count - k)
}

func quickSelect(_ nums: inout [Int], _ l: Int, _ r: Int, _ index: Int) -> Int {
    //随机一个数交换, 因为如果原来有序的数组会退化为O(n^2)
    let random = Int.random(in: l...r)
    (nums[random], nums[r]) = (nums[r], nums[random])

    //下面做分区操作，去r位置的元素作为基准值
    let x = nums[r]
    // 定义i和j指针，分为三个区域,0..<i,为小区，i..<j为大区，j..<r为未查看
    var i = l
    for j in l..<r {
        if (nums[j] <= x) {
            (nums[j], nums[i]) = (nums[i], nums[j])
            i += 1
        }
    }
    //把基准的元素放到i的位置
    (nums[r], nums[i]) = (nums[i], nums[r])
    //区别在于这里，快排提前返回
    if (i == index) {
        return nums[i]
    } else {
        return i < index ? quickSelect(&nums, i + 1, r, index) : quickSelect(&nums, l, i - 1, index)
    }
}

