import Foundation

/*
一般写算法用c语言来实现，比较能理解整个细节，因为高级语言都封装的简易的操作，
像数组是动态的，queue需要维护front和rear指针，stack要维护top指针，
c语言是面向过程的，所以在实现字符串操作相对容易，但在没有内置的map数据类型，
所以像实现LRU Cache用到双向链表和hashmap来实现，代码比较长
*/


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
//参考leetcode国际站，最简洁的代码
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
