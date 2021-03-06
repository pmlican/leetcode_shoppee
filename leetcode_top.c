
// shopee高频leetcode
//MARK: leetcode 215. 数组中的第K个最大元素 (快排提前返回)
void swap(int* a, int* b) {
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int* nums, int l, int r) {
    int i = l;
    int x = nums[r];  //取右边作为基准值
    // i 和 j指针 划分4个区域
    // 比x小  |i 比x大   |j 未查看   |r
    for (int j = l; j < r; j++) {
        //如果比
        if (nums[j] <= x) {
            //如果小于基准值x，移到左边
            swap(&nums[i], &nums[j]);
            i++;
        }
    }
    // 最后把基本值，放到i的位置，就是中间的位置
    swap(&nums[i], &nums[r]);
    return i;
}

// 随机一个partition , 防止退化成O(n^2)
int randomPartition(int *nums, int l, int r) {
    // random一个随机数，在 i - r之间
    //比如[3,2,1,|5,6,4], 随机5和4直接的值， 3(元素个数)(0-3之间) + 3(5的下标)
    //r - l + 1 = 个数， 然后加上 l, 就是大于l
    int i = rand() % (r - l + 1) + l;
    swap(&nums[i], &nums[r]);
    return partition(nums, l, r);
}

int quickSelect(int *nums, int l, int r, int index) {
    int q = randomPartition(nums, l, r);
    if (q == index) {
        return nums[q];
    } else {
        //如果 q < index， 如果index比q大，那就在右区间，递归右区间
        return q < index ? quickSelect(nums, q + 1, r, index) : quickSelect(nums, l, q - 1, index);
    }
}

int findKthLargest(int* nums, int numsSize, int k){
    srand(time(0)); //生成一个时间种子，因为随机数是根据当前时间戳生成的
    return quickSelect(nums, 0, numsSize - 1, numsSize - k);
}

//MARK: 88. 合并两个有序数组 (逆向双指针)
//逆向双指针避免数组拷贝
void merge(int* nums1, int nums1Size, int m, int* nums2, int nums2Size, int n){
    int p1 = m - 1, p2 = n - 1, tail = m + n - 1;
    int cur;
    while (p1 >= 0 || p2 >= 0) {
        if (p1 == -1) {
            cur = nums2[p2--]; //p1指针走完，nums2还有剩余
        } else if (p2 == -1) {
            cur = nums1[p1--];
        } else (nums[p1] > nums[p2]) {
            cur = nums1[p1--];
        } else {
            cur = nums2[p2--];
        }
        nums1[tail--] = cur;  //逆向防止被覆盖
    }
}


//MARK: 94. 二叉树的中序遍历
// 递归解法
void inorder(struct TreeNode* root, int* returnSize, int* res) {
    if (root == NULL) {
        return;
    }
    inorder(root->left, returnSize, res);
    res[(*returnSize)++] = root->val;
    inorder(root->right, returnSize, res);
}

int* inorderTraversal(struct TreeNode* root, int* returnSize){
    int* res = malloc(sizeof(int) * 201);
    *returnSize = 0;
    inorder(root, returnSize, res);
    return res;
}

// 用栈来模拟递归，先左，中，右
int* inorderTraversal(struct TreeNode* root, int* returnSize) {
    *returnSize = 0;
    int* res = malloc(sizeof(int) * 501);
    struct TreeNode** stk = malloc(sizeof(struct TreeNode*) * 501);
    int top = 0;
    while (root != NULL || top > 0) {
        while (root != NULL) {
            stk[top++] = root;
            root = root->left;
        }
        root = stk[--top];
        res[(*returnSize)++] = root->val;
        root = root->right;
    }
    return res;
}

//MARK: 912. 排序数组
void swap(int *x,int *y) {
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

int partition(int* a, int l, int r) {
    int x = a[r], i = l;
    for (int j = l; j < r; j++) {
        if (a[j] <= x) {
            swap(&a[i++], &a[j]);
        }
    }
    swap(&a[i], &a[r]);
    return i;
}

int randomPartition(int* a, int l, int r) {
    //random(min: l, max: r) 类似这样随机一个l到r之间的数
    //比如[3,2,1,|5,6,4], 随机5和4直接的值， 3(元素个数)(0-3之间) + 3(5的下标)
    int i = rand() % (r - l + 1) + l;
    swap(&a[i], &a[r]);
    return partition(a, l, r);
}

void quickSort(int *nums, int l, int r) {
    if (l < r) {
        int p = randomPartition(nums, l, r); //解决本身为升序数组，降为O(n^2)
        quickSort(nums, l, p - 1);
        quickSort(nums, p + 1, r);
    }
}
int* sortArray(int* nums, int numsSize, int* returnSize){
    srand(time(0));
    quickSort(nums, 0, numsSize - 1);
    *returnSize = numsSize;
    return nums;
}

//MARK: 224. 基本计算器

//因为字符只包含空格，(，)，+，-，和数字0-9

int calculate(char * s){
    int n = strlen(s); //获取字符长度
    int stack[n]; //定义一个栈，存放操作符号，如果是+ 则为+1，如果为- 则为-1 //用一个栈来存操作符，操作符 < n，保证不会溢出
    int top = 0;
    int sign = 1; //定义操作符
    stack[top++] = sign; //默认为+，并更新top, 栈顶元素下标为top - 1
    
    int ret = 0;
    int i = 0;
    while (i < n) { //遍历整个字符串
        if (s[i] == ' ') {
            //如果为空字符，什么都不处理，继续i++
            i++;
        } else if (s[i] == '+') {
            //如果是加号,取栈顶元素更新sign
            sign = stack[top - 1];
            i++;
        } else if (s[i] == '-') {
            //如果是加号,取栈顶元素更新sign为负
            sign = -stack[top - 1];
            i++;
        } else if (s[i] == '(') {
            //如果是( 入栈
            stack[top++] = sign;
            i++;
        } else if (s[i] == ')') {
            top--; //出栈是更新top指针 //操作符出栈，不用删除元素，因为维护top指针就可以，新来的值会覆盖旧值
            i++;
        } else {
            //如果是数字 i < n判断有没越界，因为如果是连续数字1233，有可能越界
            // num * 10 因为数字有可能是多位数，例如123
            //s[i] - '0' //取出当前位置数字与'0'做差值得出数字的值
            long num = 0; //因为数字可能会联系，做*10操作
            while (i < n && s[i] >= '0' && s[i] <= '9') { //i < n 防止越界，因为下面在i++
                num = num * 10 + s[i] - '0'; //s[i] - '0'计算字符ascii差值
                i++; //这里记得i++ 继续遍历
            }
            // 要乘以 ( 前面的操作符号
            ret += sign * num;
        }
    }
    return ret;
}

//MARK: 152. 乘积最大子数组
/*
 动态规划: 因为负数存在，负负得正，会导致最大的变最小，最小的变最大。所以需要维护一个最小的子数组和最大的子数组
 maxF(i) = max(maxF(i-1) * a[i], minF(i-1) * a[i], a[i])
 minF(i) = min(maxF(i-1) * a[i], minF(i-1) * a[i], a[i])
 */
//WARN: 符号优先级问题, 嵌套使用要注意: max(a, b) a > b ? a : b 不加括号会有问题 math自带的fmax也会有问题
#define max(a, b) ((a) > (b)) ? (a) : (b)
#define min(a, b) ((a) < (b)) ? (a) : (b)

int maxProduct(int* nums, int numsSize){
    int maxF[numsSize], minF[numsSize];
    memcpy(maxF, nums, sizeof(int) * numsSize);
    memcpy(minF, nums, sizeof(int) * numsSize);
    for (int i = 1; i < numsSize; i++) {
        maxF[i] = max(maxF[i - 1] * nums[i], max(nums[i], minF[i - 1] * nums[i]));
        minF[i] = min(maxF[i - 1] * nums[i], max(nums[i], minF[i - 1] * nums[i]));
//        if (nums[i] > 0) {
//            maxF[i] = fmax(maxF[i - 1] * nums[i], nums[i]);
//            minF[i] = fmin(minF[i - 1] * nums[i], nums[i]);
//        } else {
//            maxF[i] = fmax(minF[i - 1] * nums[i], nums[i]);
//            minF[i] = fmin(maxF[i - 1] * nums[i], nums[i]);
//        }
    }
    int ans = maxF[0];
    for (int i = 1; i < numsSize; i++) {
        ans = fmax(ans, maxF[i]);
    }
    return ans;
}
// 滚动数组优化空间 这里不会有符号嵌套问题，但最好自己实现max和min
int maxProduct(int* nums, int numsSize){
    int maxF = nums[0], minF = nums[0], ans = nums[0];
    for(int i = 1; i < numsSize; ++i){
        //可以用滚动数组实现，最大和最小变量,取出上一个max,和min
        int mx = maxF, mn = minF;
        maxF = fmax(mx * nums[i], fmax(nums[i], mn * nums[i]));
        minF = fmin(mn * nums[i], fmin(nums[i], mx * nums[i]));
        ans = fmax(maxF, ans);
    }
    return ans;
}

//MARK: 232. 用栈实现队列
// 一个做输入栈，一个作输出栈，均摊时间复杂度为O(1)
//每次 pop 或 peek 时，若输出栈为空则将输入栈的全部数据依次弹出并压入输出栈，这样输出栈从栈顶往栈底的顺序就是队列从队首往队尾的顺序。

typedef struct {
    int* stk; //数据域指针
    int stkSize;  //当前栈顶指针
    int stkCapacity; //栈容量
} Stack;

Stack* stackCreate(int cpacity) {
    Stack* ret = malloc(sizeof(Stack));
    ret->stk = malloc(sizeof(int) * cpacity);
    ret->stkSize = 0;
    ret->stkCapacity = cpacity;
    return ret;
}

void stackPush(Stack* obj, int x) {
    obj->stk[obj->stkSize++] = x;
}

void stackPop(Stack* obj) {
    obj->stkSize--; //出栈就是向下移动stkSize
}

int stackTop(Stack* obj) {
    return obj->stk[obj->stkSize - 1]; //返回栈顶元素
}

bool stackEmpty(Stack* obj) {
    return obj->stkSize == 0;
}

void stackFree(Stack* obj) {
    free(obj->stk);
}

typedef struct {
    Stack* inStack;
    Stack* outStack;
} MyQueue;

MyQueue* myQueueCreate() {
    MyQueue* ret = malloc(sizeof(MyQueue));
    ret->inStack = stackCreate(100);
    ret->outStack = stackCreate(100);
    return ret;
}

void in2out(MyQueue* obj) {
    while (!stackEmpty(obj->inStack)) {
        stackPush(obj->outStack, stackTop(obj->inStack)); //输入栈元素压入输出栈
        stackPop(obj->inStack); //输入栈元素出栈
    }
}

void myQueuePush(MyQueue* obj, int x) {
    stackPush(obj->inStack, x);
}

int myQueuePop(MyQueue* obj) {
    if (stackEmpty(obj->outStack)) {
        in2out(obj);  //当输出栈元素为空时，把输入栈元素压入输出栈
    }
    int x = stackTop(obj->outStack); //返回栈顶元素
    stackPop(obj->outStack); //出栈
    return x;
}

int myQueuePeek(MyQueue* obj) {
    if (stackEmpty(obj->outStack)) {
        in2out(obj); //peek的时候也要判断输出栈是否为空，因为有可能pop的时候是最后一个
    }
    return stackTop(obj->outStack);
}

bool myQueueEmpty(MyQueue* obj) {
    return stackEmpty(obj->inStack) && stackEmpty(obj->outStack);
}

void myQueueFree(MyQueue* obj) {
    stackFree(obj->inStack);
    stackFree(obj->outStack);
}

//MARK: 102. 二叉树的层序遍历
/*
 [3,9,20,null,null,15,7] ==>
      3
    /  \
   9   20
      /   \
     15   7
 [
   [3],
   [9,20],
   [15,7]
 ]
 */
/*
 1. int ** rslt 指针的指针，表示int*指针数组  int* | int*
                                        |
                                        v
                                      [int, int]
 2. 为什么不能直接int resunlt[][]?
 因为我们在函数中通过int result[][]的内存空间在程序的栈区，当被调函数结束运行后，
 栈区的内存随之被释放，该result二维数组并不能被返回到主调函数中去。而通过malloc申
 请的内存空间位于堆区，当被调函数执行结束后，可以通过指针将该二维数组传递到函数外面
 */

int** levelOrder(struct TreeNode* root, int* returnSize, int** returnColumnSizes){
    *returnSize = 0; //这句话要放在return null前面，在调用函数时，如果返回值如果是一个常量则没问题。如果返回值若为指针则可能会出现该错误，假如返回的指针地址指向函数内的局部变量，在函数退出时，该变量的存储空间会被销毁，此时去访问该地址就会出现这个错误。
    if (root == NULL) {
        return NULL;
    }
    //初始化变量操作
    int** res = (int**)malloc(sizeof(int*) * 2000);  // 二维 数组
    *returnColumnSizes = (int*)malloc(sizeof(int) * 2000);  // 一维数组：[returnColumnSizes] 列数组
    struct TreeNode* queue[2000]; //队列
    int front = 0, rear = 0; //队首和对尾指针
    struct TreeNode* cur;
    queue[rear++] = root; // root入队
    while (front != rear) { // 队列不为空
        int colSize = 0;
        int last = rear; //求当前队列的长度si
        res[*returnSize] = (int*)malloc(sizeof(int) * (last - front)); //实例化一维数组，存每一level的数据
        while(front < last) {//依次从队列中取si个元素进行拓展，然后进入下一次迭代
            cur = queue[front++]; //出队
            res[*returnSize][colSize++] = cur->val; //放入没一level的数组中
            if (cur->left != NULL) {
                queue[rear++] = cur->left; //入队
            }
            if (cur->right != NULL) {
                queue[rear++] = cur->right;
            }
        }
        //上面两个参数用来描述这个结果，以便调用者打印树的形态
        (*returnColumnSizes)[*returnSize] = colSize;  //这个参数用来“带回”每层的结点个数
        ++(*returnSize); //这个参数用来“带回”层数
    }
    return res;
}


//MARK: 69. Sqrt(x) x 的平方根
//方法一: 二分法， 扩展： 要求精确到6位数

int mySqrt(int x) {
    int l = 0, r = x, ans = -1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if ((long)mid * mid <= x) {  //long防止溢出
            ans = mid;
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return ans;
}

int mySqrt(double x, double epsilon) {
    double l = 0, r = x;
    if (x == 0 || x == 1) {
        return x;
    }
    while (left < right) {
        double mid = l + (r - l) / 2;
        if (fabs(mid * mid - x) < epsilon) {
            return mid;
        } else if (mid * mid < x) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

//牛顿迭代法 等价于 求 f(x) = x^2 - a 的正根， 因为f'(x) = 2x, 根据斜截式求出x轴交点为  x - f(x)/2x , f(x)代入得
//（x + a/x) / 2

double newtonSqrt(int a) {
    long x = a;
    while (x * x > a) {
        x = (x + a / x) / 2;
    }
    return (int)x;
}

//MARK: 912. 排序数组： 归并排序


/*
 1. 先二分数组直到元素不能再分
 2. 两两合并为一个有序的数组
 */

void merge(int* nums, int* temp, int l, int mid, int r) {
    int i = l, j = mid + 1, k = l;
    while(i <= mid && j <= r) {
        if (nums[i] > nums[j]) {
            temp[k++] = nums[j++];
        } else {
            temp[k++] = nums[i++];
        }
    }
    while (i <= mid) {
        temp[k++] = nums[i++];
    }
    while (j <= r) {
        temp[k++] = nums[j++];
    }
    //拷贝回原数组
//    memcpy(nums + l, temp, sizeof(int) * (r - l + 1));
    for (int i = l; i <= r; i++) {
        nums[i] = temp[i];
    }
}

void mergeSort(int *nums, int *temp, int l, int r) {
    if (l < r) {
//        int mid = l + (r - l) / 2;
        int mid = (l + r) >> 1;
        mergeSort(nums, temp, l, mid);
        mergeSort(nums, temp, mid + 1, r);
        merge(nums, temp, l, mid, r);
    }
}

int* sortArray(int* nums, int numsSize, int* returnSize){
    // int* temp = (int*)malloc(sizeof(int) * numsSize);
    int temp[numsSize];
    mergeSort(nums, temp, 0, numsSize - 1);
    *returnSize = numsSize;
    return nums;
}

//MARK: 41. 缺失的第一个正数
/*
 哈希表法：没有出现的最小正整数在[1, N+1]，要么[1, N]或者 N + 1

 */

int firstMissingPositive(int* nums, int numsSize) {
    //1.将所有负数改为n+1
    for (int i = 0; i < numsSize; i++) {
        if (nums[i] <= 0) {
            nums[i] = numsSize + 1;
        }
    }
    //2.将小于等于n的元素的位置变为负数
    for (int i = 0; i < numsSize; i++) {
        int num = abs(nums[i]);
        if (num <= numsSize) { //判断越界
            num[num - 1] = -abs(nums[num - 1]);
        }
    }
    //3.找出第一个大于0的元素下标加1
    for (int i = 0; i < numsSize; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }
    //否则返回n + 1
    return numsSize + 1;
}

//MARK: 14. 最长公共前缀
/*
 leets
 leetcode
 leet
 leeds
    ^
 通过纵向比较，如果第一个不相同字符则返回最长公共前缀
 */

char * longestCommonPrefix(char ** strs, int strsSize){
    if (strsSize == 0) {
        return "";
    }
    for (int i = 0; i < strlen(strs[0]); i++) { //纵向遍历字符数组
        for (in j = 0; j < strsSize; j++) {
            if (strs[j][i] != strs[0][i]) { //与的一个字符串同列字符不相同
                strs[0][i] = '\0'; //截断字符
                return strs[0];
            }
        }
    }
    return strs[0]; //全部符合则第一个字符串即为最长公共前缀
}

//MARK: 160. 相交链表
/*
 a1->a2->c1->c2->c3
          |
 b1->b2->b3
 
 
 a单独节点个数为a, b单独节点个数为b, a和b共同节点为c
 a + c + b = b + c + a
 双指针把a走完，然后移动到b
 把b走完，然后移动到a
 然后在交点交汇
 */

struct ListNode *getIntersectionNode(struct ListNode *headA, struct ListNode *headB) {
    if (headA == NULL || headB == NULL) { //边界条件判断
        return NULL;
    }
    struct ListNode *pA = headA, *pB = headB; //定义双指针
    while (pA != pB) { // 如果pA和pB不等，就意味着不是相交点或都为NULL
        pA = pA == NULL ? headB : pA->next;  // 移动pA指针，当走到尾部，移动headB
        pB = pB == NULL ? headA : pB->next; //pB指针同理
    }
    return pA; //返回pA或者pB都可以
}

//MARK: 53. 最大子数组和
#define max(a, b) ((a) > (b)) ? (a) : (b)

int maxSubArray(int* nums, int numsSize){
    //f(i) = max{f(i-1) + nums[i], nums[i]}
    int pre = 0, maxAns = nums[0];
    for(int i = 0; i < numsSize; i++) {
        pre = max(pre + nums[i], nums[i]);
        maxAns = max(maxAns, pre);
    }
    return maxAns;
}

//MARK: 206. 反转链表
// 头插法
struct ListNode* reverseList(struct ListNode* head){
    struct ListNode *pre = NULL, *cur = head;
    while (cur != NULL) {
        struct ListNode* next = cur->next;
        cur->next = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}
//递归法
struct ListNode* reverseList(struct ListNode* head){
    if (head == NULL || head->next == NULL) {
        return head;
    }
    struct ListNode *newNode = reverseList(head->next);
    head->next->next = head;
    head->next = NULL;
    return newNode;
}

//MARK: 110. 平衡二叉树

// height: 1. p为空   0
//         2. p为为空   max{height(p.left), height(p.right)} + 1
//辅助函数获取高度
int height(struct TreeNode *root) {
    if (root == NULL) {
        return 0;
    } else {
        return fmax(height(root->left), height(root->right)) + 1;
    }
}

bool isBalanced(struct TreeNode* root){
    if (root == NULL) {
        return true;
    } else {
        return fabs(height(root->left) - height(root->right)) <= 1 && isBalanced(root->left) && isBalanced(root->right);
    }
}

//MARK: 144. 二叉树的前序遍历
//辅助函数递归根左右

void preorder(struct TreeNode *root, int *res, int *resSize) {
    if (root == NULL) {
        return;
    }
    //根左右
    res[(*resSize)++] = root->val;
    preorder(root->left, res, resSize);
    preorder(root->right, res, resSize);
}

int* preorderTraversal(struct TreeNode* root, int* returnSize){
    int *res = malloc(sizeof(int) * 2000);
    *returnSize = 0;
    preorder(root, res, returnSize);
    return res;
}

//MARK: 54. 螺旋矩阵
//螺旋矩阵，转圈遍历
int* spiralOrder(int** matrix, int matrixSize, int* matrixColSize, int* returnSize){
    //边界判断
    if (matrixSize == 0 || matrixColSize[0] == 0) {
        *returnSize = 0;
        return NULL;
    }
    
    //计算元素总个数
    int rows = matrixSize, column = matrixColSize[0];
    int total = rows * column;
    int *res = malloc(sizeof(int) * total);
    *returnSize = 0;
    
    //定义左右上下边界
    int l = 0, r = column - 1, t = 0, b = rows - 1;
    while(l <= r && t <= b) {
        //左到右
        for (int i = l; i <= r; i++) {
            res[(*returnSize)++] = matrix[t][i];
        }
        //右到下
        for (int i = t + 1; i <= b ; i++) {
            res[(*returnSize)++] = matrix[i][r];
        }
        //因为边界在++或者--， 右到左和下到上，边界减小1
        if(l < r && t < b) { //不是while
            for (int i = r - 1; i > l; i--) {
                res[(*returnSize)++] = matrix[b][i];
            }
            for (int i = b; i > t; i--) {
                res[(*returnSize)++] = matrix[i][l];
            }
        }
        l++;
        r--;
        t++;
        b--;
    }
    return res;
}


//MARK: 剑指 Offer 22. 链表中倒数第k个节点
//方法一：先遍历一遍获取链表节点个数，第二次遍历判断倒数第k个输出节点
//方法二：双指针，先把快指针移动到k+1，然后快慢指针一起移动，当快指针为NULL时，慢指针刚好是倒数第k个节点

struct ListNode* getKthFromEnd(struct ListNode* head, int k){
    struct ListNode *fast = head; *slow = head;
    while(fast && k > 0) {
        fast = fast->next;
        k--;
    }
    while(fast) {
        fast = fast->next;
        slow = slow->next;
    }
    return slow;
}

//MARK: 300. 最长递增子序列

// 动态规划
/*
input  10 9 2 5 3 7 101
 dp    1  1 1 2 2 3 4
 
 dp[i] = max{dp[j]} + 1  0 <= j < i 且 nums[j] < nums[i]
 然后取 ans = max{ dp[i] }
 
 */

int lengthOfLIS(int* nums, int numsSize){
    if (numsSize == 0) {
        return 0;
    }
    int dp[numsSize]; //在栈上，但不需要返回出去，所以不需要申请堆空间的数组
    dp[0] = 1;
    int ans = 1;
    for (int i = 1; i < numsSize; i++) {
        dp[i] = 1; //赋值为1，不然里面数据为脏数据
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = fmax(dp[i], dp[j] + 1);
            }
        }
        ans = fmax(ans, dp[i]);
    }
    return ans;
}

//方法二： 贪心算法 + 二分查找，优化为nlogn
//贪心：如果要上升子序列尽可能小，那每次上升子序列最后加上的数尽可能小
/*
1. 如果 nums[i] > d[len], 则直接加入数组末尾，并更新len += 1;
2. 否则，在d数组中二分查找，找到第一个比nums[i]小的数d[k],并更新d[k+1] = nums[i];
 以输入序列 [0, 8, 4, 12, 2][0,8,4,12,2] 为例
 第一步插入 0，d = [0]
 第二步插入 8，d = [0, 8]
 第三步插入 4，d = [0, 4]
 第四步插入 12，d = [0, 4, 12]
 第五步插入 2，d = [0, 2, 12]
 */

int lengthOfLIS(int* nums, int numsSize) {
    if (numsSize == 0) {
        return 0;
    }
    int len = 1;
    int d[numsSize + 1]; // 下标从1开始，后续直接返回len不用+1操作
    d[len] = nums[0];
    for (int i = 1; i < numsSize; i++) {
        if (nums[i] > d[len]) {
            d[++len] = nums[i]; // len先+1，然后赋值
        } else {
            // 二分查找
            int l = 1, r = len, pos = 0;
            while(l <= r) {
                int mid = (l + r) >> 1;
                if (d[mid] < nums[i]) {
                    pos = mid;
                    l = mid + 1; //在右边
                } else {
                    r = mid - 1; //在左边
                }
            }
            d[pos + 1] = nums[i];
        }
    }
    return len;
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

/*
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
*/

//MARK:
